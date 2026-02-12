"""Utility helpers for part / sparse coord processing.

Contains `merge_duplicate_coords` to merge features that share the same
coordinate (or are within a tolerance grid).

This module is kept dependency-light (numpy required; torch optional).
"""

from typing import Tuple, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None
from ..pact.modules.sparse.basic import SparseTensor
from ..inference_utils import change_pcd_range
import trimesh
import json
import os


def merge_duplicated_voxels(
    coords: Union[np.ndarray, "torch.Tensor"],
    feats: Union[np.ndarray, "torch.Tensor"],
    agg: str = "sum",
    tol: float = 0.0,
    return_mapping: bool = False,
):
    """Merge features for duplicate coordinates.

    Args:
        coords: (N, D) numpy array or torch tensor of coordinates.
        feats: (N, F) or (N,) numpy array or torch tensor of features.
        agg: aggregation method: 'sum'|'mean'|'min'|'max'|'first'.
        tol: tolerance for merging. If 0.0 uses exact equality (np.unique).
             If >0, coords are quantized by rounding coords/tol.
        return_mapping: if True also return inverse mapping array of shape (N,),
                        mapping each original index to merged index.

    Returns:
        (merged_coords, merged_feats, counts) or with mapping (merged_coords, merged_feats, counts, inverse)

    Notes:
        - Output types follow input: numpy in -> numpy out; torch in -> torch out (keeps device when possible).
        - counts is always a numpy.ndarray of dtype int.
    """
    # If coords is a torch.Tensor, do a pure-torch implementation (avoid numpy roundtrip)
    if torch is not None and isinstance(coords, torch.Tensor):
        coords_t = coords
        feats_t = (
            feats
            if isinstance(feats, torch.Tensor)
            else torch.as_tensor(feats, device=coords_t.device)
        )
        device = coords_t.device

        if coords_t.ndim == 1:
            coords_t = coords_t.reshape(-1, 1)
        if feats_t.ndim == 1:
            feats_t = feats_t.reshape(-1, 1)

        N = coords_t.shape[0]
        if N == 0:
            merged_coords = torch.zeros(
                (0, coords_t.shape[1]), device=device, dtype=coords_t.dtype
            )
            merged_feats = torch.zeros(
                (0, feats_t.shape[1]), device=device, dtype=feats_t.dtype
            )
            counts = torch.zeros((0,), dtype=torch.long, device=device)
            if return_mapping:
                return (
                    merged_coords,
                    merged_feats,
                    counts,
                    torch.empty((0,), dtype=torch.long, device=device),
                )
            return merged_coords, merged_feats, counts

        if tol is not None and tol > 0:
            keys = torch.round(coords_t / float(tol)).to(torch.int64)
            uniq_keys, inverse, counts = torch.unique(
                keys, dim=0, return_inverse=True, return_counts=True
            )
            merged_coords_t = uniq_keys.to(coords_t.dtype) * float(tol)
        else:
            uniq_coords, inverse, counts = torch.unique(
                coords_t, dim=0, return_inverse=True, return_counts=True
            )
            merged_coords_t = uniq_coords

        M = merged_coords_t.shape[0]
        F = feats_t.shape[1]
        merged_feats_t = torch.zeros((M, F), dtype=feats_t.dtype, device=device)

        if agg in ("sum", "mean"):
            # index_add for sum
            merged_feats_t = merged_feats_t.index_add(0, inverse.to(device), feats_t)
            if agg == "mean":
                merged_feats_t = merged_feats_t / counts.to(merged_feats_t.dtype).view(
                    -1, 1
                )
        elif agg in ("min", "max"):
            # fallback per-group loop for min/max
            if agg == "min":
                merged_feats_t[:] = (
                    torch.finfo(feats_t.dtype).max
                    if feats_t.is_floating_point()
                    else torch.iinfo(feats_t.dtype).max
                )
                for i in range(M):
                    mask = inverse == i
                    merged_feats_t[i] = torch.min(feats_t[mask], dim=0).values
            else:
                merged_feats_t[:] = (
                    torch.finfo(feats_t.dtype).min
                    if feats_t.is_floating_point()
                    else torch.iinfo(feats_t.dtype).min
                )
                for i in range(M):
                    mask = inverse == i
                    merged_feats_t[i] = torch.max(feats_t[mask], dim=0).values
        elif agg == "first":
            first_idx = -torch.ones((M,), dtype=torch.long, device=device)
            for orig_i, g in enumerate(inverse.tolist()):
                if first_idx[g] == -1:
                    first_idx[g] = orig_i
            merged_feats_t = feats_t[first_idx]
        else:
            raise ValueError(f"Unsupported agg: {agg}")

        counts = counts.to(torch.long)
        inverse_out = inverse.to(torch.long)

        if return_mapping:
            return merged_coords_t, merged_feats_t, counts, inverse_out
        return merged_coords_t, merged_feats_t, counts

    # Fallback numpy implementation (previous behavior)
    # ...existing code...
    coords_np = np.asarray(coords)
    feats_np = np.asarray(feats)

    if coords_np.ndim == 1:
        coords_np = coords_np.reshape(-1, 1)
    if feats_np.ndim == 1:
        feats_np = feats_np.reshape(-1, 1)

    N = coords_np.shape[0]
    if N == 0:
        merged_coords = np.zeros((0, coords_np.shape[1]))
        merged_feats = np.zeros((0, feats_np.shape[1]))
        counts = np.zeros((0,), dtype=int)
        if return_mapping:
            return merged_coords, merged_feats, counts, np.array([], dtype=int)
        return merged_coords, merged_feats, counts

    if tol is not None and tol > 0:
        keys = np.round(coords_np / float(tol)).astype(np.int64)
        uniq_keys, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        merged_coords_np = uniq_keys.astype(float) * float(tol)
    else:
        uniq_coords, inverse, counts = np.unique(
            coords_np, axis=0, return_inverse=True, return_counts=True
        )
        merged_coords_np = uniq_coords.astype(coords_np.dtype)

    M = merged_coords_np.shape[0]
    F = feats_np.shape[1]
    merged_feats_np = np.zeros((M, F), dtype=feats_np.dtype)

    if agg in ("sum", "mean"):
        np.add.at(merged_feats_np, inverse, feats_np)
        if agg == "mean":
            merged_feats_np = merged_feats_np / counts.reshape(-1, 1)
    elif agg == "min":
        if np.issubdtype(feats_np.dtype, np.floating):
            fill = np.finfo(feats_np.dtype).max
        else:
            fill = np.iinfo(feats_np.dtype).max
        merged_feats_np[:] = fill
        for i in range(M):
            idx = inverse == i
            merged_feats_np[i] = np.min(feats_np[idx], axis=0)
    elif agg == "max":
        if np.issubdtype(feats_np.dtype, np.floating):
            fill = np.finfo(feats_np.dtype).min
        else:
            fill = np.iinfo(feats_np.dtype).min
        merged_feats_np[:] = fill
        for i in range(M):
            idx = inverse == i
            merged_feats_np[i] = np.max(feats_np[idx], axis=0)
    elif agg == "first":
        first_idx = np.empty(M, dtype=int)
        first_idx.fill(-1)
        for orig_i, g in enumerate(inverse):
            if first_idx[g] == -1:
                first_idx[g] = orig_i
        merged_feats_np = feats_np[first_idx]
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    counts = counts.astype(int)

    merged_coords = merged_coords_np
    merged_feats = merged_feats_np
    inverse_out = inverse

    if return_mapping:
        return merged_coords, merged_feats, counts, inverse_out
    return merged_coords, merged_feats, counts


def change_bbox_range(bbox, padding_size=2, bins=64):
    # vertex_out = (voxels + 0.5) / 64 - 0.5
    assert bbox.shape == (2, 3), f"bbox shape should be (2, 3), but got {bbox.shape}"
    # assert (bbox >= -0.5).all() and (bbox <= 0.5).all(), f"bbox values should be in [-0.5, 0.5], but got min {bbox.min()}, max {bbox.max()}"
    points = change_pcd_range(
        bbox, from_rg=(-0.5, 0.5), to_rg=(0.5 / bins, 1 - 0.5 / bins)
    )  ### NOTE::
    bbox_min = np.floor(points[0] * bins).astype(np.int32)
    bbox_max = np.ceil(points[1] * bins).astype(np.int32)
    bbox_min = np.clip(bbox_min - padding_size, 0, bins - 1)
    bbox_max = np.clip(bbox_max + padding_size, 0, bins - 1)

    return np.stack([bbox_min, bbox_max], axis=0)


def merge_multipart_slat(latent_list, full_body_part=None, verbose=False):
    ### TODO: merge multi-part slat into a full-body slat
    merged_coords = torch.tensor(
        np.concatenate([item["coords"] for item in latent_list], axis=0)
    ).to(
        torch.int32
    )  ## (N, D)
    merged_feats = torch.tensor(
        np.concatenate([item["feats"] for item in latent_list], axis=0)
    )  ## (N, C)

    if full_body_part is not None:

        pass
    else:
        mc, mf, counts, inv = merge_duplicated_voxels(
            merged_coords, merged_feats, agg="mean", tol=0.0, return_mapping=True
        )
        if verbose:
            print("mc:", mc)
            print("mf:", mf)
            print("counts:", sorted(counts))
            print("inv:", inv)
        full_body_part = SparseTensor(coords=mc, feats=mf)
        # full_body_part = SparseTensor(coords=torch.cat([torch.zeros_like(mc[:,:1]),mc],dim=-1), feats=mf)

    start_idx = full_body_part.coords.shape[0]
    part_layouts = [slice(0, full_body_part.coords.shape[0])]

    for item in latent_list:
        part_layouts.append(slice(start_idx, start_idx + item["coords"].shape[0]))
        start_idx += item["coords"].shape[0]

    coords = merged_coords
    # coords = torch.cat([torch.zeros_like(merged_coords[:, :1]), merged_coords], dim=-1)
    coords = torch.cat([full_body_part.coords, coords], dim=0)
    slat = SparseTensor(
        feats=torch.cat([full_body_part.feats, merged_feats], dim=0),
        coords=coords,
    )

    return {"slat": slat, "part_layouts": part_layouts}


def gen_mesh_from_bounds_no_transform(bounds):
    bboxes = []
    for j in range(bounds.shape[0]):
        bbox = trimesh.primitives.Box(bounds=bounds[j])
        # color = get_random_color(j, use_float=True)
        # bbox.visual.vertex_colors = color
        bboxes.append(bbox)
    mesh = trimesh.Scene(bboxes)
    # mesh.apply_transform(rot_matrix)
    return mesh


def extract_bbox_info(data):
    parts_info = []
    for item in data.get("diffuse_tree", []):
        aabb = item.get("aabb", {})
        parts_info.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "center": aabb.get("center"),
                "size": aabb.get("size"),
                "joint": item.get("joint"),
                "parent": item.get("parent"),
                "child": item.get("child"),
            }
        )

    return parts_info


def convert_to_axis_aligned_bbox(center, size):
    #
    # min_xyz = [center[i] - size[i] / 2 for i in range(3)]
    # max_xyz = [center[i] + size[i] / 2 for i in range(3)]

    c = np.array(center, dtype=np.float32)
    # c[1:] = -c[1:]  # Invert Y and Z axes
    s = np.array(size, dtype=np.float32)
    half = s * 0.5

    min_bound = (c - half).tolist()
    max_bound = (c + half).tolist()
    return {"min_xyz": min_bound, "max_xyz": max_bound}


def load_meshes(mesh_paths, base_dir=None):
    """
    Load meshes from a list of file paths using trimesh.

    Args:
        mesh_paths (list of str): List of mesh file paths.

    Returns:
        list of trimesh.Trimesh: List of loaded mesh objects.
    """
    meshes = []
    for path in mesh_paths:
        if base_dir is not None:
            path = os.path.join(base_dir, path)
        mesh = trimesh.load(path, force="mesh")
        meshes.append(mesh)
    return meshes


def build_aabb_array(parts_info, sorted_zyx=False):
    """
    Convert parts_info (list of dicts with 'center' and 'size') into an array of axis-aligned AABBs and sorted them in orders.

    Returns:
        np.ndarray: shape (N, 2, 3) where [i,0] = min, [i,1] = max. dtype=float32.
    """
    processed = []
    for b in parts_info:
        if b.get("center") is None or b.get("size") is None:
            continue
        aabb = convert_to_axis_aligned_bbox(b["center"], b["size"])
        min_xyz = np.array(aabb["min_xyz"], dtype=np.float32)
        max_xyz = np.array(aabb["max_xyz"], dtype=np.float32)
        # annotate original dict for downstream use
        b["aabb"] = {"min_xyz": min_xyz.tolist(), "max_xyz": max_xyz.tolist()}
        b["_aabb_min"] = min_xyz
        processed.append(b)

    if len(processed) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    if sorted_zyx:
        # Sort by min corner: z, then y, then x (ascending). min_xyz is (x,y,z)
        processed = sorted(
            processed,
            key=lambda b: (
                float(b["_aabb_min"][2]),
                float(b["_aabb_min"][1]),
                float(b["_aabb_min"][0]),
            ),
        )

    aabb_array = np.array(
        [[b["aabb"]["min_xyz"], b["aabb"]["max_xyz"]] for b in processed],
        dtype=np.float32,
    )
    return aabb_array, processed


def sort_meta_by_bboxes_zyx(meta_info):
    """
    Sort each bbox in bbox_list by the zyx order of bbox_min.
    Input format: [ [bbox_min, bbox_max], ... ] where bbox_min is [x, y, z].
    Output format: sorted bbox_list, in ascending order of bbox_min's (z, y, x) values.
    """
    meta_info = sorted(
        meta_info,
        key=lambda b: (
            float(b["_aabb_min"][2]),
            float(b["_aabb_min"][1]),
            float(b["_aabb_min"][0]),
        ),
    )

    aabb_array = np.array(
        [[b["aabb"]["min_xyz"], b["aabb"]["max_xyz"]] for b in meta_info],
        dtype=np.float32,
    )
    return aabb_array, meta_info


def sort_bboxes_by_zyx(bbox_list):
    """
    Sort each bbox in bbox_list by the zyx order of bbox_min.
    Input format: [ [bbox_min, bbox_max], ... ] where bbox_min is [x, y, z].
    Output format: sorted bbox_list, in ascending order of bbox_min's (z, y, x) values.
    """
    sorted_bboxes = sorted(
        bbox_list, key=lambda bbox: (bbox[0][2], bbox[0][1], bbox[0][0])
    )
    return sorted_bboxes


def prepaer_parts_meta(
    base_dir,
    part_info_path="object_merge_fixed.json",
    transform_info_path="trellis_part_preprocess/render_merged_fixed_eevee/full/transforms.json",
    verbose=False,
):

    meta_info_path = os.path.join(base_dir, part_info_path)

    with open(meta_info_path, "r") as f:
        meta_data = json.load(f)

    # bbox_npy_path = os.path.join(base_dir, f"bboxes/{idx}.npy") ## shape: (N-part, 2, 3)

    #################################### load bbox meta info Method
    transform_info_path = os.path.join(base_dir, transform_info_path)
    with open(transform_info_path, "r") as f:
        transform_data = json.load(f)
        scale = transform_data["scale"]
        translation = np.array(transform_data["offset"])

    parts_info = extract_bbox_info(meta_data)
    # build aabb_array from extracted parts_info
    bbox_aabb_array, parts_info = build_aabb_array(parts_info)
    rot_matrix_inv = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )  ## to blender formats
    # bbox_aabb_array = bbox_aabb_array * scale + translation
    mesh_bbox = gen_mesh_from_bounds_no_transform(bbox_aabb_array)
    if verbose:
        print("before transform\n\n", bbox_aabb_array)

    for name, geom in mesh_bbox.geometry.items():
        geom.apply_transform(rot_matrix_inv)
    new_bbox_list = [geom.bounds for geom in mesh_bbox.geometry.values()]

    for geom, info in zip(mesh_bbox.geometry.values(), parts_info):
        # print(f"Geometry: {info['name']}, ID: {info['id']}, New Bounds: {geom.bounds}")
        info["aabb"]["min_xyz"] = (scale * geom.bounds[0] + translation).tolist()
        info["aabb"]["max_xyz"] = (scale * geom.bounds[1] + translation).tolist()
        info["_aabb_min"] = (scale * geom.bounds[0] + translation).tolist()

    bbox_aabb_array, parts_info = sort_meta_by_bboxes_zyx(parts_info)

    # bbox_aabb_array = bbox_aabb_array * scale + translation

    #### transform axis and direction in meta info
    #### transform axis and direction in meta info
    for info in parts_info:
        info["joint"]["axis"]["origin"] = (
            scale
            * (np.array(info["joint"]["axis"]["origin"]) @ rot_matrix_inv[:3, :3].T)
            + translation
        ).tolist()
        info["joint"]["axis"]["direction"] = (
            np.array(info["joint"]["axis"]["direction"]) @ rot_matrix_inv[:3, :3].T
        ).tolist()

    if verbose:

        print("new_bbox_list\n\n", bbox_aabb_array)

        print(
            "aabb min, max, mean:",
            bbox_aabb_array.min() if bbox_aabb_array.size else None,
            bbox_aabb_array.max() if bbox_aabb_array.size else None,
            bbox_aabb_array.mean() if bbox_aabb_array.size else None,
        )
        print(bbox_aabb_array.shape)

    return parts_info, bbox_aabb_array


if __name__ == "__main__":
    # quick smoke tests when run directly
    coords = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [2, 2, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 1],
        ]
    )
    feats = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    mc, mf, counts, inv = merge_duplicated_voxels(
        coords, feats, agg="sum", tol=0.0, return_mapping=True
    )
    print("mc:", mc)
    print("mf:", mf)
    print("counts:", counts)
    print("inv:", inv)
