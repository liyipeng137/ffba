import numpy as np
import os, cv2
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt


def visualize_chunks(chunks, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Assign each chunk a random color 
    xyz_by_chunks, rgb_by_chunks = [], []
    for i, a_chunk in enumerate(chunks):
        color = torch.rand(3).view(1,3)
        xyz_by_chunks.append(a_chunk.reshape(-1,3))
        rgb_by_chunks.append((color.expand_as(a_chunk.reshape(-1,3))))
    xyz_by_chunks = torch.cat(xyz_by_chunks, dim=0)
    rgb_by_chunks = torch.cat(rgb_by_chunks, dim=0)
    save_xyzrgb_to_ply(xyz_by_chunks, rgb_by_chunks, output_file)

def save_images_as_grid(images, filename, num_per_row=2):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save images    (N, C, H, W) or (N, H, W, C)
    if type(images) == torch.Tensor:
        if images.shape[1] == 3:
            show_img = images.permute(0, 2, 3, 1) #B,H,W,C
        elif images.shape[-1] == 3:
            show_img = images
        show_img = show_img.cpu().numpy() #N,H,W,C
        show_img = (show_img*255).astype(np.uint8)
    elif type(images) == np.ndarray:
        if images.shape[1] == 3:
            show_img = images.transpose(0, 2, 3, 1)
        elif images.shape[-1] == 3:
            show_img = images
    else:
        raise TypeError(f'Unsupported type {type(images)} for saving images.')
    # arrange images in a grid
    N = show_img.shape[0]
    num_row = int(np.ceil(N / num_per_row))
    height, width = show_img.shape[1:3]
    grid_img = np.ones((num_row * height, num_per_row * width, 3), dtype=np.uint8)*255
    for i in range(N):
        row = i // num_per_row
        col = i % num_per_row
        grid_img[row * height:(row + 1) * height, col * width:(col + 1) * width] = show_img[i]
    # save the grid image
    cv2.imwrite(filename, grid_img[..., ::-1])
    return 

def create_error_map(x, min_val=None, max_val=None, cmap='jet'):
    if type(x) == torch.Tensor:
        x = x.cpu().numpy()
    x = np.clip(x, min_val, max_val)
    x = (x - min_val) / (max_val - min_val) 
    x = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_JET)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    error_bar = draw_errorbar(min_val, max_val, width=300, height=100, cmap=cmap, unit='mm')
    return x.squeeze(1), error_bar


def draw_errorbar(min_val, max_val, width=256, height=50, cmap='jet', unit='cm'):
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor('white')

    # Adjust the axes position to control thickness and margins
    ax.set_position([0.1, 0.25, 0.8, 0.5])

    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap(cmap)

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal'
    )

    # Move the label above the bar and tighten spacing
    cb.set_label(f'Error ({unit})', fontsize=16, labelpad=-25, fontweight='regular')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.tick_params(labelsize=16, width=1)

    plt.tight_layout()

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img


def save_xyzrgb_to_ply(points, rgb, filename):
    points = points.reshape(-1,3)
    rgb = rgb.reshape(-1,3)
    
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except:
        pass
    # Save points
    if type(points) == torch.Tensor:
        points = points.cpu().numpy()
    if type(rgb) == torch.Tensor:
        rgb = rgb.cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
            
    assert points.shape[0] == rgb.shape[0]
    assert points.shape[1] == 3
    assert rgb.shape[1] == 3
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            point = points[i]
            color = rgb[i]
            f.write(f'{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n')