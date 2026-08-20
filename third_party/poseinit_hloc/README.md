# poseinit_hloc

Standalone Nerfstudio transforms.json + HLOC initialization path extracted from gaustudio.

Expected input:

- `source_path/transforms.json`
- images referenced by each frame's `file_path`

Run:

```bash
python -m poseinit_hloc \
  --source_path /path/to/input \
  --output_dir /path/to/output \
  --resolution 1.0 \
  --overwrite
```

Outputs:

- `output_dir/images`
- `output_dir/sparse/0`
- `output_dir/sparse/0/points3D.ply`

This module keeps `camera_id=1` and writes one global PINHOLE camera. If per-frame intrinsics are present in `transforms.json`, the cached per-frame intrinsics are averaged into that global camera. All cached images must have the same width and height.
