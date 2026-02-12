import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import cv2
import torch
import trimesh
import glob
from tqdm import tqdm


def load_img_mask(
    img_path, mask_path, size=(518, 518), img=None, mask_part=None, is_sort=True
):
    if img_path is not None:
        image = Image.open(img_path)
    else:
        image = img
    alpha = np.array(image.getchannel(3))
    bbox = np.array(alpha).nonzero()
    bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center_offset = [0, 0]
    aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
    aug_bbox = [
        int(aug_center[0] - aug_hsize),
        int(aug_center[1] - aug_hsize),
        int(aug_center[0] + aug_hsize),
        int(aug_center[1] + aug_hsize),
    ]
    img_height, img_width = alpha.shape

    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    else:
        mask = mask_part

    pad_left = max(0, -aug_bbox[0])
    pad_top = max(0, -aug_bbox[1])
    pad_right = max(0, aug_bbox[2] - img_width)
    pad_bottom = max(0, aug_bbox[3] - img_height)

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    # print("mask unique",torch.unique(torch.from_numpy(mask)))
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_array = np.array(image)
        padded_img_array = np.pad(
            img_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        # print("[INFO] Image padded due to bbox exceeding image boundaries."
        #   )
        # print(mask.shape, aug_bbox, pad_left, pad_top, pad_right, pad_bottom)
        mask_pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if mask.ndim > 2:
            mask_pad_width.extend([(0, 0)] * (mask.ndim - 2))
        padded_mask_array = np.pad(
            mask, tuple(mask_pad_width), mode="constant", constant_values=0
        )
        image = Image.fromarray(padded_img_array.astype("uint8"))
        aug_bbox[0] += pad_left
        aug_bbox[1] += pad_top
        aug_bbox[2] += pad_left
        aug_bbox[3] += pad_top
        mask = padded_mask_array

    image = image.crop(aug_bbox)
    mask = mask[aug_bbox[1] : aug_bbox[3], aug_bbox[0] : aug_bbox[2]]
    ordered_mask_input, mask_vis = load_bottom_up_mask(
        mask, size=size, mask_size=(37, 37), is_sort=is_sort
    )

    image_white_bg = np.array(image)
    image_black_bg = np.array(image)
    if image_white_bg.shape[-1] == 4:
        mask_img = image_white_bg[..., 3] == 0
        image_white_bg[mask_img] = [255, 255, 255, 255]
        image_black_bg[mask_img] = [0, 0, 0, 255]
        image_white_bg = image_white_bg[..., :3]
        image_black_bg = image_black_bg[..., :3]
        img_white_bg = Image.fromarray(image_white_bg.astype("uint8"))
        img_black_bg = Image.fromarray(image_black_bg.astype("uint8"))

    img_white_bg = img_white_bg.resize(size, resample=Image.Resampling.LANCZOS)
    img_black_bg = img_black_bg.resize(size, resample=Image.Resampling.LANCZOS)
    img_mask_vis = vis_mask_on_img(img_white_bg, mask_vis)
    img_white_bg = TF.to_tensor(img_white_bg)
    img_black_bg = TF.to_tensor(img_black_bg)

    return img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis


def load_bottom_up_mask(mask, size=(518, 518), mask_size=(37, 37), is_sort=True):
    mask_input = smart_downsample_mask(mask, mask_size)
    mask_vis = cv2.resize(mask_input, size, interpolation=cv2.INTER_NEAREST)
    mask_input = np.array(mask_input, dtype=np.int32)

    if is_sort:
        unique_indices = np.unique(mask_input)
        unique_indices = unique_indices[unique_indices > 0]

        part_positions = {}
        for idx in unique_indices:
            y_coords, _ = np.where(mask_input == idx)
            if len(y_coords) > 0:
                part_positions[idx] = np.max(y_coords)

        sorted_parts = sorted(
            part_positions.items(), key=lambda x: -x[1]
        )  # Sort by y-coordinate in descending order
        # Create mapping from old indices to new indices (ordered by position)
        index_map = {}
        for new_idx, (old_idx, _) in enumerate(
            sorted_parts, 1
        ):  # Start from 1 (0 is background)
            index_map[old_idx] = new_idx
        # Apply the mapping to create position-ordered mask
        ordered_mask_input = np.zeros_like(mask_input)
        for old_idx, new_idx in index_map.items():
            ordered_mask_input[mask_input == old_idx] = new_idx
    else:
        ordered_mask_input = mask_input
    mask_vis = np.array(mask_vis, dtype=np.int32)
    ordered_mask_input = torch.from_numpy(ordered_mask_input).long()

    return ordered_mask_input, mask_vis


def smart_downsample_mask(mask, target_size):
    h, w = mask.shape[:2]
    target_h, target_w = target_size
    h_ratio = h / target_h
    w_ratio = w / target_w

    downsampled = np.zeros((target_h, target_w), dtype=mask.dtype)
    for i in range(target_h):
        for j in range(target_w):
            y_start = int(i * h_ratio)
            y_end = min(int((i + 1) * h_ratio), h)
            x_start = int(j * w_ratio)
            x_end = min(int((j + 1) * w_ratio), w)
            region = mask[y_start:y_end, x_start:x_end]
            if region.size == 0:
                continue
            unique_values, counts = np.unique(region.flatten(), return_counts=True)
            non_zero_mask = unique_values > 0
            if np.any(non_zero_mask):
                non_zero_values = unique_values[non_zero_mask]
                non_zero_counts = counts[non_zero_mask]
                max_idx = np.argmax(non_zero_counts)
                downsampled[i, j] = non_zero_values[max_idx]
            else:
                max_idx = np.argmax(counts)
                downsampled[i, j] = unique_values[max_idx]

    return downsampled


def vis_mask_on_img(img, mask):
    H, W = mask.shape
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8) + 255
    for part_id in range(1, int(mask.max()) + 1):
        part_mask = mask == part_id
        if part_mask.sum() > 0:
            color = get_random_color((part_id - 1), use_float=False)[:3]
            mask_vis[part_mask, 0:3] = color
    mask_img = Image.fromarray(mask_vis)
    combined_width = W * 2
    combined_height = H
    combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    combined_img.paste(img, (0, 0))
    combined_img.paste(mask_img, (W, 0))
    draw = ImageDraw.Draw(combined_img)
    draw.line([(W, 0), (W, H)], fill=(0, 0, 0), width=2)

    return combined_img


def get_random_color(index: Optional[int] = None, use_float: bool = False):
    # some pleasing colors
    # matplotlib.colormaps['Set3'].colors + matplotlib.colormaps['Set2'].colors + matplotlib.colormaps['Set1'].colors
    palette = np.array(
        [
            [141, 211, 199, 255],
            [255, 255, 179, 255],
            [190, 186, 218, 255],
            [251, 128, 114, 255],
            [128, 177, 211, 255],
            [253, 180, 98, 255],
            [179, 222, 105, 255],
            [252, 205, 229, 255],
            [217, 217, 217, 255],
            [188, 128, 189, 255],
            [204, 235, 197, 255],
            [255, 237, 111, 255],
            [102, 194, 165, 255],
            [252, 141, 98, 255],
            [141, 160, 203, 255],
            [231, 138, 195, 255],
            [166, 216, 84, 255],
            [255, 217, 47, 255],
            [229, 196, 148, 255],
            [179, 179, 179, 255],
            [228, 26, 28, 255],
            [55, 126, 184, 255],
            [77, 175, 74, 255],
            [152, 78, 163, 255],
            [255, 127, 0, 255],
            [255, 255, 51, 255],
            [166, 86, 40, 255],
            [247, 129, 191, 255],
            [153, 153, 153, 255],
        ],
        dtype=np.uint8,
    )

    if index is None:
        index = np.random.randint(0, len(palette))

    if index >= len(palette):
        index = index % len(palette)

    if use_float:
        return palette[index].astype(np.float32) / 255
    else:
        return palette[index]


def change_pcd_range(pcd, from_rg=(-1, 1), to_rg=(-1, 1)):
    pcd = (pcd - (from_rg[0] + from_rg[1]) / 2) / (from_rg[1] - from_rg[0]) * (
        to_rg[1] - to_rg[0]
    ) + (to_rg[0] + to_rg[1]) / 2
    return pcd


def prepare_bbox_gen_input(
    voxel_coords_path, img_white_bg, ordered_mask_input, bins=64, device="cuda"
):
    whole_voxel = np.load(voxel_coords_path)
    whole_voxel = whole_voxel[:, 1:]
    whole_voxel = (whole_voxel + 0.5) / bins - 0.5
    whole_voxel_index = change_pcd_range(
        whole_voxel, from_rg=(-0.5, 0.5), to_rg=(0.5 / bins, 1 - 0.5 / bins)
    )
    whole_voxel_index = (whole_voxel_index * bins).astype(np.int32)

    points = torch.from_numpy(whole_voxel).to(torch.float16).unsqueeze(0).to(device)
    whole_voxel_index = (
        torch.from_numpy(whole_voxel_index).long().unsqueeze(0).to(device)
    )
    images = img_white_bg.unsqueeze(0).to(device)
    masks = ordered_mask_input.unsqueeze(0).to(device)

    return {
        "points": points,
        "whole_voxel_index": whole_voxel_index,
        "images": images,
        "masks": masks,
    }


def vis_voxel_coords(voxel_coords, bins=64):
    voxel_coords = voxel_coords[:, 1:]
    voxel_coords = (voxel_coords + 0.5) / bins - 0.5
    voxel_coords_ply = trimesh.PointCloud(voxel_coords)
    rot_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    voxel_coords_ply.apply_transform(
        rot_matrix
    )  ### NOTE:: TODO:Why need this? maybe because of the different coordinate system between our voxel and the ply viewer?
    return voxel_coords_ply


def gen_mesh_from_bounds(bounds):
    bboxes = []
    rot_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for j in range(bounds.shape[0]):
        bbox = trimesh.primitives.Box(bounds=bounds[j])
        color = get_random_color(j, use_float=True)
        bbox.visual.vertex_colors = color
        bboxes.append(bbox)
    mesh = trimesh.Scene(bboxes)
    mesh.apply_transform(rot_matrix)
    return mesh


def merge_parts(save_dir):
    scene_list = []
    scene_list_texture = []
    part_list = glob.glob(os.path.join(save_dir, "*.glb"))
    part_list = [
        p for p in part_list if "part" in p and "parts" not in p and "part0" not in p
    ]  # part 0 is the overall model
    part_list.sort()
    for i, part_path in enumerate(tqdm(part_list, desc="Merging parts")):
        part_mesh = trimesh.load(part_path, force="mesh")
        scene_list_texture.append(part_mesh)

        random_color = get_random_color(i, use_float=True)
        part_mesh_color = part_mesh.copy()
        part_mesh_color.visual = trimesh.visual.ColorVisuals(
            mesh=part_mesh_color, vertex_colors=random_color
        )
        scene_list.append(part_mesh_color)
        os.remove(part_path)
    scene_texture = trimesh.Scene(scene_list_texture)
    scene_texture.export(os.path.join(save_dir, "mesh_textured.glb"))
    scene = trimesh.Scene(scene_list)
    scene.export(os.path.join(save_dir, "mesh_segment.glb"))
