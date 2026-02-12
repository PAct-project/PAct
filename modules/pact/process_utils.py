import os
import imageio
import torch
from modules.pact.utils import render_utils, postprocessing_utils
from modules.pact.representations.gaussian.gaussian_model import Gaussian


def change_pcd_range(pcd, from_rg=(-1, 1), to_rg=(-1, 1)):
    pcd = (pcd - (from_rg[0] + from_rg[1]) / 2) / (from_rg[1] - from_rg[0]) * (
        to_rg[1] - to_rg[0]
    ) + (to_rg[0] + to_rg[1]) / 2
    return pcd


def merge_gaussians(gaussians_list):
    if not gaussians_list:
        raise ValueError("gaussians_list is empty")

    first_gaussian = gaussians_list[0]
    merged_gaussian = Gaussian(
        **first_gaussian.init_params, device=first_gaussian.device
    )

    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []

    for gaussian in gaussians_list:
        if gaussian.sh_degree != first_gaussian.sh_degree or not torch.allclose(
            gaussian.aabb, first_gaussian.aabb
        ):
            raise ValueError(
                "All Gaussian objects must have the same sh_degree and aabb parameters"
            )

        if gaussian._xyz is not None:
            xyz_list.append(gaussian._xyz)
        if gaussian._features_dc is not None:
            features_dc_list.append(gaussian._features_dc)
        if gaussian._features_rest is not None:
            features_rest_list.append(gaussian._features_rest)
        if gaussian._scaling is not None:
            scaling_list.append(gaussian._scaling)
        if gaussian._rotation is not None:
            rotation_list.append(gaussian._rotation)
        if gaussian._opacity is not None:
            opacity_list.append(gaussian._opacity)

    if xyz_list:
        merged_gaussian._xyz = torch.cat(xyz_list, dim=0)
    if features_dc_list:
        merged_gaussian._features_dc = torch.cat(features_dc_list, dim=0)
    if features_rest_list:
        merged_gaussian._features_rest = torch.cat(features_rest_list, dim=0)
    else:
        merged_gaussian._features_rest = None
    if scaling_list:
        merged_gaussian._scaling = torch.cat(scaling_list, dim=0)
    if rotation_list:
        merged_gaussian._rotation = torch.cat(rotation_list, dim=0)
    if opacity_list:
        merged_gaussian._opacity = torch.cat(opacity_list, dim=0)

    return merged_gaussian


def exploded_coords(coords, explosion_scale=0.4):
    if not coords:
        raise ValueError("coords list is empty")

    centers = []
    for coord in coords:
        if coord.ndim < 2 or coord.size(-1) != 3:
            raise ValueError("Each coord tensor must have shape [*, 3]")
        centers.append(
            coord.mean(dim=0)
            if coord.dtype == torch.float32
            else coord.float().mean(dim=0)
        )

    centers = torch.stack(centers)
    global_center = centers.mean(dim=0)

    exploded_coords_list = []
    for coord, part_center in zip(coords, centers):
        direction = part_center - global_center
        direction_norm = torch.norm(direction)
        if direction_norm <= 1e-6:
            # Fall back to a random unit direction if the part center matches the global center
            direction = torch.randn_like(direction)
            direction = direction / torch.norm(direction)
        else:
            direction = direction / direction_norm

        offset = direction * explosion_scale
        exploded_coords_list.append(coord + offset)

    return torch.cat(exploded_coords_list, dim=0)


def exploded_gaussians(gaussians_list, explosion_scale=0.4):

    if not gaussians_list:
        raise ValueError("gaussians_list is empty")

    first_gaussian = gaussians_list[0]
    merged_gaussian = Gaussian(
        **first_gaussian.init_params, device=first_gaussian.device
    )

    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []

    all_centers = []
    for gaussian in gaussians_list:
        if gaussian._xyz is not None:
            center = gaussian.get_xyz.mean(dim=0)
            all_centers.append(center)

    if not all_centers:
        raise ValueError("No valid gaussians with xyz data found")

    all_centers = torch.stack(all_centers)
    global_center = all_centers.mean(dim=0)

    for i, gaussian in enumerate(gaussians_list):
        if gaussian.sh_degree != first_gaussian.sh_degree or not torch.allclose(
            gaussian.aabb, first_gaussian.aabb
        ):
            raise ValueError(
                "All Gaussian objects must have the same sh_degree and aabb parameters"
            )

        if i < len(all_centers):
            part_center = all_centers[i]
            direction = part_center - global_center
            direction_norm = torch.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            else:
                direction = torch.randn(3, device=gaussian.device)
                direction = direction / torch.norm(direction)

            offset = direction * explosion_scale
        else:
            offset = torch.zeros(3, device=gaussian.device)

        if gaussian._xyz is not None:
            original_xyz = gaussian.get_xyz
            exploded_xyz = original_xyz + offset
            exploded_xyz_normalized = (
                exploded_xyz - gaussian.aabb[None, :3]
            ) / gaussian.aabb[None, 3:]
            xyz_list.append(exploded_xyz_normalized)

        if gaussian._features_dc is not None:
            features_dc_list.append(gaussian._features_dc)
        if gaussian._features_rest is not None:
            features_rest_list.append(gaussian._features_rest)
        if gaussian._scaling is not None:
            scaling_list.append(gaussian._scaling)
        if gaussian._rotation is not None:
            rotation_list.append(gaussian._rotation)
        if gaussian._opacity is not None:
            opacity_list.append(gaussian._opacity)

    if xyz_list:
        merged_gaussian._xyz = torch.cat(xyz_list, dim=0)
    if features_dc_list:
        merged_gaussian._features_dc = torch.cat(features_dc_list, dim=0)
    if features_rest_list:
        merged_gaussian._features_rest = torch.cat(features_rest_list, dim=0)
    else:
        merged_gaussian._features_rest = None
    if scaling_list:
        merged_gaussian._scaling = torch.cat(scaling_list, dim=0)
    if rotation_list:
        merged_gaussian._rotation = torch.cat(rotation_list, dim=0)
    if opacity_list:
        merged_gaussian._opacity = torch.cat(opacity_list, dim=0)

    return merged_gaussian


def make_slat_coords_from_voxel_coords(voxel_coords, num_parts):

    predefine_empty_part = (
        torch.zeros((4, 3)).cuda()
        + torch.arange(0, 4).unsqueeze(1).repeat(1, 3).cuda()
        + 59
    )
    objs = []
    objs_sorted_by_bbox = []
    bbox_sorted = []
    slat_data_list = []
    start = 0
    for i in range(len(num_parts)):
        objs.append(voxel_coords[start : start + num_parts[i]])
        start += num_parts[i]
    for obj_idx in range(len(objs)):
        parts_data = []
        for part_idx in range(len(objs[obj_idx])):
            part_voxel_coords = objs[obj_idx][part_idx]

            if part_voxel_coords.shape[0] == 0:
                print("\033[91m" "## Skipping." "\033[0m")
                print(
                    "Empty part detected, adding default:",
                    part_idx,
                    "of obj_idx:",
                    obj_idx,
                )
                # print("Empty part detected, skipping part_idx:",part_idx,"of obj_idx:",obj_idx)
                # continue
                part_voxel_coords = predefine_empty_part
            min_xyz = part_voxel_coords.min(0)[0]
            max_xyz = part_voxel_coords.max(0)[0]
            bbox = torch.stack([min_xyz, max_xyz], 0)

            parts_data.append(
                {
                    "bbox": bbox,
                    "voxel_coords": part_voxel_coords,
                }
            )
        bbox_sorted.append([part["bbox"] for part in parts_data])
        sorted_parts_data = sorted(
            parts_data,
            key=lambda x: (x["bbox"][0, 2], x["bbox"][0, 1], x["bbox"][0, 0]),
        )
        overall_voxel_coords = None
        for sorted_part in sorted_parts_data:
            if overall_voxel_coords is None:
                overall_voxel_coords = sorted_part["voxel_coords"]
            else:
                overall_voxel_coords = torch.cat(
                    [overall_voxel_coords, sorted_part["voxel_coords"]], 0
                )
        objs_sorted_by_bbox.append(sorted_parts_data)
        part_layouts = []
        ######################
        start_idx = 0
        part_layouts.append(slice(start_idx, start_idx + overall_voxel_coords.shape[0]))
        start_idx += overall_voxel_coords.shape[0]
        for part_idx in range(len(sorted_parts_data)):
            part_layouts.append(
                slice(
                    start_idx,
                    start_idx + sorted_parts_data[part_idx]["voxel_coords"].shape[0],
                )
            )
            start_idx += sorted_parts_data[part_idx]["voxel_coords"].shape[0]
        coords_obj_idx = torch.cat([overall_voxel_coords, overall_voxel_coords], dim=0)
        slat_data_list.append(
            {
                "coords": torch.cat(
                    [
                        torch.zeros(
                            coords_obj_idx.shape[0],
                            1,
                            dtype=torch.int32,
                            device=coords_obj_idx.device,
                        ),
                        coords_obj_idx,
                    ],
                    dim=-1,
                ).to(torch.int32),
                "part_layouts": part_layouts,
            }
        )
        #######################

    return objs_sorted_by_bbox, bbox_sorted, slat_data_list


def merge_multi_parts(x_0, num_parts, explode_ratio=0.0):
    s = 0
    merged_x0 = []
    predefine_empty_part = (
        torch.zeros((4, 3)).cuda()
        + torch.arange(0, 4).unsqueeze(1).repeat(1, 3).cuda()
        + 1
    )
    for num_part in num_parts:
        x_0_part = x_0[s : s + num_part]
        # filter empty parts

        x_0_part = [
            part if part.shape[0] > 0 else predefine_empty_part for part in x_0_part
        ]
        if explode_ratio > 0.0:
            coord_rg = (0, 64)
            x_0_part = [
                change_pcd_range(x_0_part_i, from_rg=coord_rg, to_rg=(1e-3, 1 - 1e-3))
                for x_0_part_i in x_0_part
            ]
            x_0_part = exploded_coords(x_0_part, explosion_scale=explode_ratio)

        else:
            x_0_part = torch.cat(x_0_part, dim=0).cuda()
        s += num_part
        merged_x0.append(x_0_part)

    return merged_x0
