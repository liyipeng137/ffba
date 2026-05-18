import argparse
import torch
import cv2
import numpy as np
from mdm.model.v2 import MDMModel
from pathlib import Path


def load_depth_map_npy(depth_path, target_size=(1400, 1904)):
    depth_map = np.load(depth_path).astype(np.float32)
    depth_map = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_NEAREST)
    return np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)


def depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255
    ).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_colored[~valid_mask] = [0, 0, 0]
    return depth_colored

# 1348.8440565673804 1352.169146314345 700 952
parser = argparse.ArgumentParser()
parser.add_argument("images_dir")
parser.add_argument("depth_dir")
parser.add_argument("output_dir")
parser.add_argument("--fx", type=float, default=1348.8440565673804)
parser.add_argument("--fy", type=float, default=1352.169146314345)
parser.add_argument("--cx", type=float, default=700)
parser.add_argument("--cy", type=float, default=952)
args = parser.parse_args()

images_dir = Path(args.images_dir)
depth_dir = Path(args.depth_dir)
output_dir = Path(args.output_dir)

out_npy = output_dir / "depth_npy"
out_vis = output_dir / "depth_vis"
out_png = output_dir / "depth_png"
for d in [out_npy, out_vis, out_png]:
    d.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14-v0.5').to(device)

img_files = sorted(images_dir.glob("*.jpg"))
print(f"Found {len(img_files)} images")

for img_path in img_files:
    stem = img_path.stem
    depth_path = depth_dir / f"{stem}.npy"
    if not depth_path.exists():
        print(f"  [skip] no depth for {stem}")
        continue

    image_bgr = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    image_t = torch.tensor(image_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]

    depth_np = load_depth_map_npy(depth_path)
    depth_t = torch.tensor(depth_np, dtype=torch.float32, device=device)[None]

    intrinsics = np.array([
        [args.fx / w,         0, args.cx / w],
        [          0, args.fy / h, args.cy / h],
        [          0,           0,           1],
    ])
    intrinsics_t = torch.tensor(intrinsics, dtype=torch.float32, device=device)[None]

    with torch.no_grad():
        output = model.infer(
            image_t,
            depth_in=depth_t,
            enable_depth_mask=False,
            intrinsics=intrinsics_t,
        )

    depth_pred = output['depth'].squeeze().cpu().numpy()

    # 1. float32 npy
    np.save(out_npy / f"{stem}.npy", depth_pred)

    # 2. depth vis
    depth_colored = depth_to_color_opencv(depth_pred)
    cv2.imwrite(str(out_vis / f"{stem}.png"), depth_colored)

    # 3. uint16 png
    depth_clean = np.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth_u16 = np.clip(depth_clean * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(out_png / f"{stem}.png"), depth_u16)

    print(f"  ✓ {stem}")

print(f"\nDone -> {output_dir}")
