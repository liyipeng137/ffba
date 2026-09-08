import os
import torch.nn.functional as F
import gc
import torch


def mnn_one_to_many(A, B, tau=0.6):
    """
    A: (P, D)  normalized patch tokens (image i)
    B: (Bsz, P, D) normalized patch tokens (candidate images)
    tau: cosine threshold

    Returns:
        scores: (Bsz,) MNN score for each candidate in B
    """
    if B.dim() == 2:
        B = B.unsqueeze(0)  # (1, P, D)

    S = torch.matmul(B, A.t())  # (Bsz, P, P)

    j_best = S.argmax(dim=2)  # (Bsz, P)

    i_best = S.argmax(dim=1)  # (Bsz, P)

    P = A.shape[0]
    i_idx = (
        torch.arange(P, device=A.device).view(1, P).expand(B.shape[0], P)
    )  # (Bsz, P)

    i_back = torch.gather(i_best, dim=1, index=j_best)  # (Bsz, P)
    mutual = i_back == i_idx

    sim_ij = torch.gather(S, dim=2, index=j_best.unsqueeze(2)).squeeze(2)  # (Bsz, P)
    confident = sim_ij > tau

    good = mutual & confident
    return good.float().mean(dim=1)


def mnn_from_dino_candidates(
    X, sim_dino, K=30, tau=0.6, batch_cand=8, use_fp16=True, symmetric=True
):
    """
    X: (M, P, D) patch tokens
    sim_dino: (M, M) DINO similarity matrix (larger = more similar)
    K: number of candidates per image to verify with MNN
    tau: patch-level cosine threshold for confident MNN matches
    batch_cand: compute MNN for this many candidates at once
    symmetric: mirror scores to make S_mnn symmetric

    Returns:
        S_mnn: (M, M) float32 matrix with only candidate entries filled (others 0)
        cand_idx: (M, K) candidate indices used per row
    """
    device = X.device
    M, P, D = X.shape
    K = min(int(K), max(M - 1, 0))
    if K == 0:
        return (
            torch.zeros((M, M), device=device, dtype=torch.float32),
            torch.empty((M, 0), device=device, dtype=torch.long),
        )

    X = F.normalize(X, p=2, dim=-1)

    sim = sim_dino.clone()
    sim.fill_diagonal_(-1e9)

    cand_vals, cand_idx = torch.topk(
        sim, k=K, dim=1, largest=True, sorted=True
    )  # (M,K)

    S_mnn = torch.zeros((M, M), device=device, dtype=torch.float32)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_fp16 and device.type == "cuda")
        else torch.cpu.amp.autocast(enabled=False)
    )

    for i in range(M):
        A = X[i]  # (P,D)
        js = cand_idx[i]  # (K,)

        if symmetric:
            js = js[js > i]
            if js.numel() == 0:
                continue

        for t in range(0, js.numel(), batch_cand):
            j_batch = js[t : t + batch_cand]
            B = X[j_batch]  # (Bsz,P,D)

            with autocast_ctx:
                scores = mnn_one_to_many(A, B, tau=tau)  # (Bsz,)

            S_mnn[i, j_batch] = scores.float()

    if symmetric:
        S_mnn = torch.maximum(S_mnn, S_mnn.t())  # keep larger of two directions
        S_mnn.fill_diagonal_(0.0)

    return S_mnn, cand_idx


def get_sim_matrix(
    images,
    model_name="dinov3",
    device="cuda",
    subset_size=100,
    feature_size=768,
    alpha=0.3,
    return_feats=False,
):
    """Extract normalized feature embeddings for all images.

    Loads a DINOv2/DINOv3 model, normalizes images using
    ImageNet statistics, and obtains per-frame patch embeddings.
    Similarity is computed as a blend of:
    - Global mean-patch cosine similarity (diagonal zeroed).
    - MNN (mutual nearest-neighbor) patch consistency.

    """
    _RESNET_MEAN = [0.485, 0.456, 0.406]
    _RESNET_STD = [0.229, 0.224, 0.225]
    use_hf_dinov3 = model_name == "dinov3"
    try:
        from transformers import AutoModel

        model_id = os.environ.get(
            "MERG3R_DINOV3_MODEL_ID",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
        )
        model = AutoModel.from_pretrained(model_id)
    except Exception as e:
        print(f"[UTILS] Error loading DINOv3 model: {e}")
        raise e

    model.eval()
    model = model.to(device)

    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    num_subsets = (len(images) + subset_size - 1) // subset_size
    frame_feat = torch.empty(size=(0, feature_size), device=device)
    frame_chunks = []
    non_blocking = images.device.type == "cpu" and images.is_pinned()
    with torch.no_grad():
        for i in range(num_subsets):
            image_subset = images[i * subset_size : (i + 1) * subset_size]
            if image_subset.shape[0] == 0:
                continue
            image_subset = image_subset.to(device, non_blocking=non_blocking)
            image_subset = (image_subset - resnet_mean) / resnet_std

            if use_hf_dinov3:
                outputs = model(image_subset)
                num_register_tokens = getattr(model.config, "num_register_tokens", 0)
                frame_feat_subset = outputs.last_hidden_state[
                    :, 1 + num_register_tokens :, :
                ]
            else:
                frame_feat_subset = model(image_subset, is_training=True)

                frame_feat_subset = frame_feat_subset["x_norm_patchtokens"]

            frame_chunks.append(frame_feat_subset)
            del image_subset, frame_feat_subset
    frame_feat = torch.cat(frame_chunks, dim=0)
    frame_feat_norm = torch.nn.functional.normalize(frame_feat, p=2, dim=-1)
    frame_feat_norm_mean = torch.mean(frame_feat_norm, dim=1)

    del model, resnet_mean, resnet_std
    sim_matrix = frame_feat_norm_mean @ frame_feat_norm_mean.t()
    sim_matrix = sim_matrix.fill_diagonal_(0)

    mnn_sim_matrix, cand_idx = mnn_from_dino_candidates(frame_feat_norm, sim_matrix)
    sim_matrix = alpha * sim_matrix + (1 - alpha) * mnn_sim_matrix

    gc.collect()
    torch.cuda.empty_cache()
    if return_feats:
        return sim_matrix, frame_feat_norm_mean
    return sim_matrix
