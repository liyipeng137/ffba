import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import DisjointSet

from algos.geometry import project_3d_points_to_image_numpy
from algos.tracking_metrics import filter_tracks_by_reprojection, print_tracking_metrics
from algos.utils import get_sim_matrix


def _make_loma_config(arch: str):
    from loma.loma import LoMaB, LoMaB128, LoMaG, LoMaL, LoMaR

    arch_to_cfg = {
        "LoMa-B": LoMaB,
        "LoMa-B128": LoMaB128,
        "LoMa-L": LoMaL,
        "LoMa-G": LoMaG,
        "LoMa-R": LoMaR,
    }
    if arch not in arch_to_cfg:
        raise ValueError(f"Unknown LoMA architecture: {arch}")
    return arch_to_cfg[arch]()


def _build_loma_model(arch: str, device: str):
    from loma.loma import LoMa

    model = LoMa(_make_loma_config(arch)).eval()
    return model.to(device)


def _loma_to_pixel_coords(keypoints, height: int, width: int):
    return torch.stack(
        (
            width * (keypoints[..., 0] + 1) / 2,
            height * (keypoints[..., 1] + 1) / 2,
        ),
        dim=-1,
    )


def _preprocess_loma_descriptor_image(image, height=784, width=784):
    image = F.interpolate(
        image,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    return image[None]


@torch.no_grad()
def _extract_loma_features(model, image, max_num_keypoints: int, device: str):
    image = image.to(device)
    batch = {"image": image}
    height, width = image.shape[2:]

    detections = model._detector.detect(batch, num_keypoints=max_num_keypoints)
    keypoints_norm = detections["keypoints"]
    descriptions = model._descriptor.describe_keypoints(
        _preprocess_loma_descriptor_image(image).to(device),
        keypoints_norm,
    )["descriptions"]

    keypoints = _loma_to_pixel_coords(keypoints_norm, height, width)
    keypoints = keypoints - 0.5
    keypoints[..., 0] = keypoints[..., 0].clamp(0.5, width - 1.5)
    keypoints[..., 1] = keypoints[..., 1].clamp(0.5, height - 1.5)

    return {
        "keypoints": keypoints[0],
        "keypoints_norm": keypoints_norm,
        "descriptors": descriptions,
        "scores": detections.get("keypoint_probs", None),
    }


@torch.no_grad()
def _match_loma_features(model, feats0, feats1, filter_threshold: float):
    from loma.loma import filter_matches

    scores = model(
        feats0["keypoints_norm"],
        feats1["keypoints_norm"],
        feats0["descriptors"],
        feats1["descriptors"],
    )["scores"]
    m0, _, _, _ = filter_matches(scores, filter_threshold)
    valid = m0[0] > -1
    if not torch.any(valid):
        return torch.empty((0, 2), dtype=torch.long, device=m0.device)

    match_indices0 = torch.where(valid)[0]
    match_indices1 = m0[0][valid]
    return torch.stack([match_indices0, match_indices1], dim=-1)


def _graph_pairs_from_similarity(images, k: int):
    sim_matrix = get_sim_matrix(images)
    num_images = images.shape[0]
    if num_images < 2:
        return []

    effective_k = min(k, num_images - 1)
    pairs = []
    seen_pairs = set()
    for i in range(num_images):
        sim_row = sim_matrix[i].clone()
        if num_images - i - 1 >= effective_k:
            indices = torch.arange(0, num_images, device=sim_row.device)
            sim_row[indices <= i] = -1
        else:
            sim_row[i] = -1

        top_k_neighbors = torch.topk(sim_row, effective_k)
        for n in top_k_neighbors.indices:
            j = int(n.item())
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pairs.append((i, j))
    return pairs


def _valid_geometric_matches(
    matches,
    kpts0,
    kpts1,
    points0,
    points1,
    extrinsic0,
    extrinsic1,
    intrinsic0,
    intrinsic1,
    max_reproj_error: float,
):
    if matches.shape[0] == 0:
        return matches

    m_kpts0 = kpts0[matches[..., 0]].cpu().numpy()
    m_kpts1 = kpts1[matches[..., 1]].cpu().numpy()

    height0, width0 = points0.shape[:2]
    height1, width1 = points1.shape[:2]
    m_kpts0_round = m_kpts0.round().astype(int)
    m_kpts1_round = m_kpts1.round().astype(int)
    m_kpts0_round[..., 0] = np.clip(m_kpts0_round[..., 0], 0, width0 - 1)
    m_kpts0_round[..., 1] = np.clip(m_kpts0_round[..., 1], 0, height0 - 1)
    m_kpts1_round[..., 0] = np.clip(m_kpts1_round[..., 0], 0, width1 - 1)
    m_kpts1_round[..., 1] = np.clip(m_kpts1_round[..., 1], 0, height1 - 1)

    query_points_3d_0 = points0[m_kpts0_round[..., 1], m_kpts0_round[..., 0]]
    query_points_3d_1 = points1[m_kpts1_round[..., 1], m_kpts1_round[..., 0]]

    reproj_pixel_0_on_1, valid_mask_0 = project_3d_points_to_image_numpy(
        query_points_3d_0,
        extrinsic1[:3, :3],
        extrinsic1[:3, 3:],
        intrinsic1,
    )
    reproj_pixel_1_on_0, valid_mask_1 = project_3d_points_to_image_numpy(
        query_points_3d_1,
        extrinsic0[:3, :3],
        extrinsic0[:3, 3:],
        intrinsic0,
    )

    error0 = np.linalg.norm(reproj_pixel_0_on_1 - m_kpts1, axis=-1)
    error1 = np.linalg.norm(reproj_pixel_1_on_0 - m_kpts0, axis=-1)
    valid_matches = (
        (error0 < max_reproj_error)
        & (error1 < max_reproj_error)
        & valid_mask_0
        & valid_mask_1
    )
    return matches[valid_matches]


def _build_tracks_from_matches(all_matches, all_features, points, depth_conf, num_images):
    final_track = [[] for _ in range(num_images)]
    points_id = [[] for _ in range(num_images)]
    final_points = []
    final_points_conf = []

    for subset in all_matches.subsets():
        curr_point_id = len(final_points)
        point = np.zeros((3), dtype=np.float64)
        conf_sum = 0.0
        seen_images = set()
        observations = []

        for img, point_id in subset:
            if img in seen_images:
                continue
            seen_images.add(img)

            query_point = all_features[img]["keypoints"].squeeze()[point_id]
            query_point_np = query_point.cpu().numpy()

            height, width = points[img].shape[:2]
            query_points_round = query_point_np.round().astype(int)
            x = int(np.clip(query_points_round[0], 0, width - 1))
            y = int(np.clip(query_points_round[1], 0, height - 1))
            query_points_3d = points[img][y, x]
            query_points_3d_conf = float(np.asarray(depth_conf[img][y, x]).reshape(-1)[0])

            point += query_points_3d * query_points_3d_conf
            conf_sum += query_points_3d_conf
            observations.append((img, query_point_np))

        if len(seen_images) < 2 or np.any(conf_sum <= 0):
            continue

        for img, query_point_np in observations:
            final_track[img].append(query_point_np)
            points_id[img].append(curr_point_id)

        final_points.append(point / conf_sum)
        final_points_conf.append(conf_sum / len(seen_images))

    final_track = [np.stack(track) if track else np.array([]) for track in final_track]
    points_id = [np.stack(idx) if idx else np.array([]) for idx in points_id]

    if not final_points:
        return (
            final_track,
            points_id,
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )

    final_points = np.stack(final_points).astype(np.float32)
    final_points_conf = np.stack(final_points_conf).astype(np.float32)
    return final_track, points_id, final_points, final_points_conf


@torch.no_grad()
def graph_extract_matches_loma(
    images,
    points,
    depth_conf,
    extrinsic,
    intrinsic,
    k=10,
    sim_thresh=0.9,
    max_num_keypoints=4096,
    max_reproj_error=8.0,
    device="cuda",
    arch="LoMa-B",
    filter_threshold=0.1,
):
    """
    LoMA graph tracking for MERG3R.

    Pair selection, bidirectional 3D reprojection filtering, and DisjointSet track
    merging follow graph_extract_matches_lightglue. LoMA only replaces the local
    feature extraction and pairwise matching backend.
    """
    del sim_thresh  # Kept for signature parity with graph_extract_matches_lightglue.

    model = _build_loma_model(arch=arch, device=device)
    num_images = images.shape[0]
    pairs = _graph_pairs_from_similarity(images, k=k)

    all_features = []
    all_matches = DisjointSet()
    for i in range(num_images):
        all_features.append(
            _extract_loma_features(
                model,
                images[i : i + 1],
                max_num_keypoints=max_num_keypoints,
                device=device,
            )
        )

    for i0, i1 in pairs:
        matches = _match_loma_features(
            model,
            all_features[i0],
            all_features[i1],
            filter_threshold=filter_threshold,
        )
        matches = _valid_geometric_matches(
            matches,
            all_features[i0]["keypoints"],
            all_features[i1]["keypoints"],
            points[i0],
            points[i1],
            extrinsic[i0],
            extrinsic[i1],
            intrinsic[i0],
            intrinsic[i1],
            max_reproj_error=max_reproj_error,
        )

        for j in range(matches.shape[0]):
            pt0 = (i0, matches[j, 0].cpu().item())
            pt1 = (i1, matches[j, 1].cpu().item())
            all_matches.add(pt0)
            all_matches.add(pt1)
            all_matches.merge(pt0, pt1)

    print("End LoMA tracking.")
    print("Num of LoMA graph pairs: ", len(pairs))
    track, points_id, points_3d, points_conf = _build_tracks_from_matches(
        all_matches,
        all_features,
        points,
        depth_conf,
        num_images,
    )
    print_tracking_metrics(
        "LoMAGraphBeforePostFilter",
        track,
        points_id,
        points_3d,
        extrinsic,
        intrinsic,
        points_conf=points_conf,
    )
    track, points_id, points_3d, points_conf = filter_tracks_by_reprojection(
        track,
        points_id,
        points_3d,
        points_conf,
        extrinsic,
        intrinsic,
        max_reproj_error=max_reproj_error,
        min_track_length=2,
        label="LoMAGraph",
    )
    print_tracking_metrics(
        "LoMAGraph",
        track,
        points_id,
        points_3d,
        extrinsic,
        intrinsic,
        points_conf=points_conf,
        extra_stats={
            "pair_count": len(pairs),
            "arch": arch,
            "filter_threshold": filter_threshold,
            "max_reproj_error": max_reproj_error,
        },
    )
    return track, points_id, points_3d, points_conf
