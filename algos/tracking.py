import torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import DisjointSet

from algos.feedforward_paths import ensure_feedforward_on_path

ensure_feedforward_on_path()

from algos.geometry import project_3d_points_to_image_numpy
from algos.tracking_metrics import filter_tracks_by_reprojection, print_tracking_metrics
from algos.utils import get_sim_matrix, rbd
from lightglue import ALIKED, SIFT, LightGlue, SuperPoint


def initialize_feature_extractors(max_query_num, det_thres=0.005, extractor_method="aliked", device="cuda"):
    extractors = {}
    methods = extractor_method.lower().split("+")

    for method in methods:
        method = method.strip()
        if method == "aliked":
            extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
            extractors["aliked"] = extractor.to(device).eval()
        elif method == "sp":
            extractor = SuperPoint(max_num_keypoints=max_query_num, detection_threshold=det_thres)
            extractors["sp"] = extractor.to(device).eval()
        elif method == "sift":
            extractor = SIFT(max_num_keypoints=max_query_num)
            extractors["sift"] = extractor.to(device).eval()
        else:
            print(f"Warning: Unknown feature extractor '{method}', ignoring.")

    if not extractors:
        print(f"Warning: No valid extractors found in '{extractor_method}'. Using ALIKED by default.")
        extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
        extractors["aliked"] = extractor.to(device).eval()

    return extractors


def extract_keypoints(query_image, extractors, round_keypoints=True):
    query_points = None

    with torch.no_grad():
        for extractor in extractors.values():
            query_points_data = extractor.extract(query_image, invalid_mask=None)
            extractor_points = query_points_data["keypoints"]
            if round_keypoints:
                extractor_points = extractor_points.round()

            if query_points is not None:
                query_points = torch.cat([query_points, extractor_points], dim=1)
            else:
                query_points = extractor_points

    return query_points


def select_query_points_all_images(images, extractor_method="aliked+sp", max_query_points=1024, det_threshold=0.005):
    query_points = []
    extractors = initialize_feature_extractors(max_query_num=max_query_points, extractor_method=extractor_method, 
                                               det_thres=det_threshold)

    for image in images:
        query_points.append(extract_keypoints(image, extractors, round_keypoints=False))

    return query_points


def generate_similarity_matrix(frame_feat, spatial_similarity=False):
     # Process features based on similarity type
    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

        # Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))

    distance_matrix = 100 - similarity_matrix.clone()

    
    return similarity_matrix, distance_matrix



@torch.no_grad()
def extract_matches_lightglue(images, points, depth_conf, extrinsic, intrinsic, steps=[1], max_num_keypoints=4096, max_reproj_error=8.0, device='cuda'):
    """
    images: (N, 3, H, W)
    points: (N, H, W, 3)
    depth_conf: (N, H, W, 1)
    extrinsic: (N, 3, 4)
    intrinsic: (N, 3, 3)

    Return:
    tracks: list of (P_i, 2) for each image i
    points_id: list of (P_i) for each image i
    points: (P, 3)
    points_conf: (P, 1)
    """
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    N = images.shape[0]

    all_features = []
    all_matches = DisjointSet()

    for i in range(images.shape[0]):
        all_features.append(extractor.extract(images[i].to(device)))


    # Skip connections
    for step in steps:
        for i in range(0, len(images) - step):
            feats0 = all_features[i]
            feats1 = all_features[i + step]

            matches01 = matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches01 = [
                rbd(x) for x in [feats0, feats1, matches01]
            ]  # remove batch dimension
            matches = matches01["matches"]  # (N, 2)
            kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"],
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()

            # Geometric verification
            m_kpts0_round_long = m_kpts0.round().astype(int)
            m_kpts1_round_long = m_kpts1.round().astype(int)
            query_points_3d_1 = points[i][
                m_kpts0_round_long[..., 1], m_kpts0_round_long[..., 0]
            ]
            query_points_3d_2 = points[i + step][
                m_kpts1_round_long[..., 1], m_kpts1_round_long[..., 0]
            ]

            reproj_pixel_1_on_2, valid_mask_1 = project_3d_points_to_image_numpy(query_points_3d_1, extrinsic[i + step, :3, :3], extrinsic[i + step, :3, 3:], intrinsic[i + step])
            reproj_pixel_2_on_1, valid_mask_2 = project_3d_points_to_image_numpy(query_points_3d_2, extrinsic[i, :3, :3], extrinsic[i, :3, 3:], intrinsic[i])

            error1 = np.linalg.norm(reproj_pixel_1_on_2 - m_kpts1, axis=-1)
            error2 = np.linalg.norm(reproj_pixel_2_on_1 - m_kpts0, axis=-1)

            valid_matches = (error1 < max_reproj_error) & (error2 < max_reproj_error) & valid_mask_1 & valid_mask_2

            matches = matches[valid_matches]

            for j in range(matches.shape[0]):
                pt1 = (i, matches[j, 0].cpu().item())
                pt2 = (i + step, matches[j, 1].cpu().item())
                all_matches.add(pt1)
                all_matches.add(pt2)
                all_matches.merge(pt1, pt2)
        
    # Randomly delete redundant frames
    # Optionally only keep track with more than 2 images
    print("End tracking.")
    final_track = [[] for _ in range(N)]
    points_id = [[] for _ in range(N)]
    final_points = []
    final_points_conf = []
    for subset in all_matches.subsets():
        curr_point_id = len(final_points)
        point = np.zeros((3))
        conf_sum = 0
        seen_images = set()
        observations = []
        for img, point_id in subset:
            if img in seen_images:
                continue
            seen_images.add(img)
            query_point = all_features[img]['keypoints'].squeeze()[point_id]
            query_point_np = query_point.cpu().numpy()

            query_points_round_long = query_point.squeeze(0).long().cpu().numpy()
            query_points_3d = points[img][
                query_points_round_long[1], query_points_round_long[0]
            ]
            query_points_3d_conf = depth_conf[img][
                query_points_round_long[1], query_points_round_long[0]
            ]

            point += query_points_3d * query_points_3d_conf
            conf_sum += query_points_3d_conf
            observations.append((img, query_point_np))

        if len(seen_images) < 2 or np.any(conf_sum <= 0):
            continue
        point = point / conf_sum
        conf = conf_sum / len(seen_images)

        for img, query_point_np in observations:
            final_track[img].append(query_point_np)
            points_id[img].append(curr_point_id)

        final_points.append(point)
        final_points_conf.append(conf)
            
    # final_track = [np.stack(track) for track in final_track]
    # points_id = [np.stack(idx) for idx in points_id]

    final_track = [np.stack(track) if track else np.array([]) for track in final_track]
    points_id = [np.stack(idx) if idx else np.array([]) for idx in points_id ]
    
    if final_points:
        final_points = np.stack(final_points)
        final_points_conf = np.stack(final_points_conf)
    else:
        final_points = np.empty((0, 3), dtype=np.float32)
        final_points_conf = np.empty((0,), dtype=np.float32)

    return final_track, points_id, final_points, final_points_conf


@torch.no_grad()
def graph_extract_matches_lightglue(images, points, depth_conf, extrinsic, intrinsic, k=10, sim_thresh=0.9, max_num_keypoints=4096, max_reproj_error=8.0, device='cuda'):
    """
    images: (N, 3, H, W)
    points: (N, H, W, 3)
    depth_conf: (N, H, W, 1)
    extrinsic: (N, 3, 4)
    intrinsic: (N, 3, 3)
    """
    sim_matrix = get_sim_matrix(images)
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    N = images.shape[0]

    all_features = []
    all_matches = DisjointSet()

    for i in range(images.shape[0]):
        all_features.append(extractor.extract(images[i].to(device)))
    pairs = []

    for i in range(N):
        if N - i - 1 >= k:
            indices = torch.arange(0, N)
            sim_row = sim_matrix[i].clone() 
            sim_row[indices <= i] = -1
            top_k_neighbors = torch.topk(sim_row, k)
            top_k_neighbors_indices = top_k_neighbors.indices
        else:
            top_k_neighbors = torch.topk(sim_matrix[i], k)
            top_k_neighbors_indices = (top_k_neighbors.indices).to(device=device, dtype=torch.int16)
        

        for n in top_k_neighbors_indices:
            pairs.append((i, n.item()))

    raw_match_total = 0
    valid_match_total = 0
    raw_match_counts = []
    valid_match_counts = []
    valid_pair_reproj_errors = []

    for i1, i2 in pairs:
        feats0 = all_features[i1]
        feats1 = all_features[i2]

        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        matches = matches01["matches"]  # (N, 2)
        raw_match_count = int(matches.shape[0])
        raw_match_total += raw_match_count
        raw_match_counts.append(raw_match_count)
        kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"],
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()

        # Geometric verification
        m_kpts0_round_long = m_kpts0.round().astype(int)
        m_kpts1_round_long = m_kpts1.round().astype(int)
        query_points_3d_1 = points[i1][
            m_kpts0_round_long[..., 1], m_kpts0_round_long[..., 0]
        ]
        query_points_3d_2 = points[i2][
            m_kpts1_round_long[..., 1], m_kpts1_round_long[..., 0]
        ]

        reproj_pixel_1_on_2, valid_mask_1 = project_3d_points_to_image_numpy(query_points_3d_1, extrinsic[i2, :3, :3], extrinsic[i2, :3, 3:], intrinsic[i2])
        reproj_pixel_2_on_1, valid_mask_2 = project_3d_points_to_image_numpy(query_points_3d_2, extrinsic[i1, :3, :3], extrinsic[i1, :3, 3:], intrinsic[i1])

        error1 = np.linalg.norm(reproj_pixel_1_on_2 - m_kpts1, axis=-1)
        error2 = np.linalg.norm(reproj_pixel_2_on_1 - m_kpts0, axis=-1)

        valid_matches = (error1 < max_reproj_error) & (error2 < max_reproj_error) & valid_mask_1 & valid_mask_2
        # print("Num of Valid Matches: ", valid_matches.sum())
        valid_match_count = int(valid_matches.sum())
        valid_match_total += valid_match_count
        valid_match_counts.append(valid_match_count)
        if valid_match_count > 0:
            valid_pair_reproj_errors.append(np.maximum(error1[valid_matches], error2[valid_matches]))
        matches = matches[valid_matches]

        for j in range(matches.shape[0]):
            pt1 = (i1, matches[j, 0].cpu().item())
            pt2 = (i2, matches[j, 1].cpu().item())
            all_matches.add(pt1)
            all_matches.add(pt2)
            all_matches.merge(pt1, pt2)
            
    # Randomly delete redundant frames
    # Optionally only keep track with more than 2 images
    print("End tracking.")
    final_track = [[] for _ in range(N)]
    points_id = [[] for _ in range(N)]
    final_points = []
    final_points_conf = []
    for subset in all_matches.subsets():
        curr_point_id = len(final_points)
        point = np.zeros((3))
        conf_sum = 0
        seen_images = set()
        observations = []
        for img, point_id in subset:
            if img in seen_images:
                continue
            seen_images.add(img)
            query_point = all_features[img]['keypoints'].squeeze()[point_id]
            query_point_np = query_point.cpu().numpy()

            query_points_round_long = query_point.squeeze(0).long().cpu().numpy()
            query_points_3d = points[img][
                query_points_round_long[1], query_points_round_long[0]
            ]
            query_points_3d_conf = depth_conf[img][
                query_points_round_long[1], query_points_round_long[0]
            ]

            point += query_points_3d * query_points_3d_conf
            conf_sum += query_points_3d_conf
            observations.append((img, query_point_np))

        if len(seen_images) < 2 or np.any(conf_sum <= 0):
            continue
        point = point / conf_sum
        conf = conf_sum / len(seen_images)

        for img, query_point_np in observations:
            final_track[img].append(query_point_np)
            points_id[img].append(curr_point_id)

        final_points.append(point)
        final_points_conf.append(conf)
            
    # final_track = [np.stack(track) for track in final_track if track != []]
    final_track = [np.stack(track) if track else np.array([]) for track in final_track]
    print("Num of Final Track: ", len(final_track))

    # points_id = [np.stack(idx) for idx in points_id if idx != []]
    points_id = [np.stack(idx) if idx else np.array([]) for idx in points_id ]


    if final_points:
        final_points = np.stack(final_points)
        final_points_conf = np.stack(final_points_conf)
    else:
        final_points = np.empty((0, 3), dtype=np.float32)
        final_points_conf = np.empty((0,), dtype=np.float32)

    print_tracking_metrics(
        "LightGlueGraphBeforePostFilter",
        final_track,
        points_id,
        final_points,
        extrinsic,
        intrinsic,
        points_conf=final_points_conf,
    )
    final_track, points_id, final_points, final_points_conf = filter_tracks_by_reprojection(
        final_track,
        points_id,
        final_points,
        final_points_conf,
        extrinsic,
        intrinsic,
        max_reproj_error=max_reproj_error,
        min_track_length=2,
        label="LightGlueGraph",
    )

    valid_pair_error_values = (
        np.concatenate(valid_pair_reproj_errors)
        if valid_pair_reproj_errors
        else np.empty((0,), dtype=np.float32)
    )
    extra_stats = {
        "pair_count": len(pairs),
        "raw_matches": raw_match_total,
        "valid_matches_after_depth_reproj_filter": valid_match_total,
        "valid_match_ratio": f"{valid_match_total / max(raw_match_total, 1):.3f}",
        "raw_matches_per_pair": (
            "n=0"
            if not raw_match_counts
            else f"median={np.median(raw_match_counts):.3f}, mean={np.mean(raw_match_counts):.3f}, max={np.max(raw_match_counts):.3f}"
        ),
        "valid_matches_per_pair": (
            "n=0"
            if not valid_match_counts
            else f"median={np.median(valid_match_counts):.3f}, mean={np.mean(valid_match_counts):.3f}, max={np.max(valid_match_counts):.3f}"
        ),
        "valid_pair_filter_reproj_px": (
            "n=0"
            if valid_pair_error_values.size == 0
            else (
                f"median={np.median(valid_pair_error_values):.3f}, "
                f"mean={np.mean(valid_pair_error_values):.3f}, "
                f"p90={np.percentile(valid_pair_error_values, 90):.3f}, "
                f"max={np.max(valid_pair_error_values):.3f}"
            )
        ),
    }
    print_tracking_metrics(
        "LightGlueGraph",
        final_track,
        points_id,
        final_points,
        extrinsic,
        intrinsic,
        points_conf=final_points_conf,
        extra_stats=extra_stats,
    )

    return final_track, points_id, final_points, final_points_conf
