"""reporting / statistics for the formal SIFT + prior + BAE pipeline."""

import numpy as np


def summarize_counts(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0, "zero": 0}
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "zero": int(np.sum(values == 0)),
    }


def format_count_summary(label, values):
    summary = summarize_counts(values)
    return (
        f"{label}: min={summary['min']}, median={summary['median']:.1f}, "
        f"mean={summary['mean']:.1f}, max={summary['max']}, "
        f"zero={summary['zero']}"
    )


def summarize_intrinsics(initial_intrinsics, averaged_intrinsics, camera_model):
    initial_intrinsics = np.asarray(initial_intrinsics, dtype=np.float64)
    averaged_intrinsics = np.asarray(averaged_intrinsics, dtype=np.float64)
    initial_fx = initial_intrinsics[:, 0, 0]
    initial_fy = initial_intrinsics[:, 1, 1]
    initial_cx = initial_intrinsics[:, 0, 2]
    initial_cy = initial_intrinsics[:, 1, 2]
    averaged_k = averaged_intrinsics[0]

    delta = averaged_intrinsics - initial_intrinsics
    return {
        "method": "gluemap.intrinsics_averaging",
        "camera_model": camera_model,
        "num_images": int(initial_intrinsics.shape[0]),
        "intrinsics_mapping": "shared",
        "initial": {
            "fx_min": float(initial_fx.min()),
            "fx_median": float(np.median(initial_fx)),
            "fx_max": float(initial_fx.max()),
            "fy_min": float(initial_fy.min()),
            "fy_median": float(np.median(initial_fy)),
            "fy_max": float(initial_fy.max()),
            "cx_median": float(np.median(initial_cx)),
            "cy_median": float(np.median(initial_cy)),
        },
        "averaged": {
            "fx": float(averaged_k[0, 0]),
            "fy": float(averaged_k[1, 1]),
            "cx": float(averaged_k[0, 2]),
            "cy": float(averaged_k[1, 2]),
        },
        "delta_abs": {
            "fx_mean": float(np.mean(np.abs(delta[:, 0, 0]))),
            "fx_max": float(np.max(np.abs(delta[:, 0, 0]))),
            "fy_mean": float(np.mean(np.abs(delta[:, 1, 1]))),
            "fy_max": float(np.max(np.abs(delta[:, 1, 1]))),
            "cx_mean": float(np.mean(np.abs(delta[:, 0, 2]))),
            "cx_max": float(np.max(np.abs(delta[:, 0, 2]))),
            "cy_mean": float(np.mean(np.abs(delta[:, 1, 2]))),
            "cy_max": float(np.max(np.abs(delta[:, 1, 2]))),
        },
    }


def save_intrinsics_artifacts(
    output_dir,
    initial_intrinsics,
    averaged_intrinsics,
    intrinsics_mapping,
    image_names,
):
    mapping_array = np.asarray(
        [intrinsics_mapping[idx] for idx in range(len(intrinsics_mapping))],
        dtype=np.int64,
    )
    np.savez_compressed(
        output_dir / "intrinsics_refine_inputs.npz",
        initial_intrinsics=initial_intrinsics.astype(np.float64, copy=False),
        averaged_intrinsics=averaged_intrinsics.astype(np.float64, copy=False),
        shared_global_intrinsic=averaged_intrinsics[0].astype(np.float64, copy=False),
        intrinsics_mapping=mapping_array,
        image_names=np.asarray(image_names),
    )


def summarize_numeric(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }
