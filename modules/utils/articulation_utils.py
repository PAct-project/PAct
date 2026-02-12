import imageio
import itertools
import numpy as np
import torch
from modules.pact.process_utils import merge_gaussians
from modules.pact.representations.gaussian.gaussian_model import Gaussian
from modules.pact.representations.mesh import MeshExtractResult
from modules.pact.utils import render_utils
from typing import Dict, List, Union

# reference of object categories
cat_ref = {
    "Table": 0,
    "Dishwasher": 1,
    "StorageFurniture": 2,
    "Refrigerator": 3,
    "WashingMachine": 4,
    "Microwave": 5,
    "Oven": 6,
}

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 0,
        "drawer": 1,
        "base": 2,
        "handle": 3,
        "wheel": 4,
        "knob": 5,
        "shelf": 6,
        "tray": 7,
    },
    "bwd": {
        0: "door",
        1: "drawer",
        2: "base",
        3: "handle",
        4: "wheel",
        5: "knob",
        6: "shelf",
        7: "tray",
    },
}

# reference of joint types for each part
joint_ref = {
    "fwd": {"fixed": 1, "revolute": 2, "prismatic": 3, "screw": 4, "continuous": 5},
    "bwd": {1: "fixed", 2: "revolute", 3: "prismatic", 4: "screw", 5: "continuous"},
}


def post_process_arti_info(
    arti_info: List[dict],
    reps_parts: List[Gaussian],
    is_axis_o_projection: bool = True,
) -> List[dict]:
    """Augment articulation metadata with bounding boxes and optional axis projection.

    Args:
        arti_info: List of articulation nodes produced by the regressor/decoder.
        reps_parts: Geometric representations (Gaussians) corresponding to each part.
        is_axis_o_projection: If ``True`` the joint axis origin is snapped to the
            closest surface point on the part's bounding box.

    Returns:
        The input ``arti_info`` list with updated ``aabb`` entries (center, size,
        min, max) and optionally adjusted axis origins.
    """

    if not arti_info or not reps_parts:
        return arti_info

    bbox_stats: List[Dict[str, np.ndarray]] = []
    for rep in reps_parts:
        if hasattr(rep, "get_xyz"):
            xyz = rep.get_xyz  ## gaussian splatting rep
        else:
            xyz = rep.vertices  ## mesh reprentation
        if xyz is None or xyz.numel() == 0:
            min_xyz = max_xyz = torch.zeros(
                3, device=xyz.device if xyz is not None else "cpu"
            )
        else:
            min_xyz = torch.min(xyz, dim=0)[0]
            max_xyz = torch.max(xyz, dim=0)[0]
        min_np = min_xyz.detach().cpu().numpy().astype(np.float32)
        max_np = max_xyz.detach().cpu().numpy().astype(np.float32)
        size_np = np.maximum(max_np - min_np, 1e-6)
        center_np = (min_np + max_np) * 0.5
        bbox_stats.append(
            {
                "center": center_np,
                "size": size_np,
                "min": min_np,
                "max": max_np,
            }
        )

    def _project_to_bbox_surface(
        point: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, axis_dir: np.ndarray
    ) -> np.ndarray:
        point = np.asarray(point, dtype=np.float32)
        axis_dir = np.asarray(axis_dir, dtype=np.float32)
        norm = np.linalg.norm(axis_dir)
        if norm < 1e-6:
            return np.clip(point, bmin, bmax)
        axis_dir = axis_dir / norm

        intersections = []
        eps = 1e-6
        dims = [0, 1, 2]
        for axis in dims:
            other_dims = [d for d in dims if d != axis]
            for combo in itertools.product([0, 1], repeat=2):
                start = np.zeros(3, dtype=np.float32)
                for idx_other, dim in enumerate(other_dims):
                    start[dim] = bmin[dim] if combo[idx_other] == 0 else bmax[dim]
                start[axis] = bmin[axis]
                end = start.copy()
                end[axis] = bmax[axis]
                direction = end - start
                denom = float(np.dot(axis_dir, direction))
                if abs(denom) < eps:
                    plane_value = np.dot(axis_dir, point - start)
                    if abs(plane_value) < eps:
                        intersections.extend([start.copy(), end.copy()])
                    continue
                t = float(np.dot(axis_dir, point - start) / denom)
                if -eps <= t <= 1.0 + eps:
                    t_clamped = min(max(t, 0.0), 1.0)
                    intersections.append(start + t_clamped * direction)

        if not intersections:
            return np.clip(point, bmin, bmax)

        distances = [
            np.linalg.norm(intersection - point) for intersection in intersections
        ]
        closest_idx = int(np.argmin(distances))
        closest_point = intersections[closest_idx]
        return np.clip(closest_point, bmin, bmax)

    node_indices = list(range(len(arti_info)))
    if len(node_indices) != len(bbox_stats):
        non_base_indices = [
            i for i, node in enumerate(arti_info) if node.get("name") != "base"
        ]
        if len(non_base_indices) == len(bbox_stats):
            node_indices = non_base_indices
        else:
            node_indices = node_indices[: min(len(node_indices), len(bbox_stats))]

    for part_idx, node_idx in enumerate(node_indices):
        node = arti_info[node_idx]
        stats = bbox_stats[part_idx]
        aabb = node.setdefault("aabb", {})
        aabb["center"] = stats["center"].tolist()
        aabb["size"] = stats["size"].tolist()
        aabb["min"] = stats["min"].tolist()
        aabb["max"] = stats["max"].tolist()
        ###
        if not is_axis_o_projection or node.get("joint", {}).get("type") != "revolute":
            # print("not revolute")
            continue

        joint = node.get("joint") or {}
        axis_info = joint.get("axis") or {}
        origin = axis_info.get("origin")
        if origin is None:
            continue
        axis_direction = axis_info.get("direction")
        if axis_direction is None:
            continue
        projected = _project_to_bbox_surface(
            origin, stats["min"], stats["max"], axis_direction
        )
        axis_info["origin"] = projected.tolist()
        joint["axis"] = axis_info
        node["joint"] = joint

    return arti_info


def _clone_gaussian(rep: Gaussian) -> Gaussian:
    cloned = Gaussian(**rep.init_params, device=rep.device)
    cloned._xyz = rep._xyz.clone() if rep._xyz is not None else None
    cloned._features_dc = (
        rep._features_dc.clone() if rep._features_dc is not None else None
    )
    cloned._features_rest = (
        rep._features_rest.clone() if rep._features_rest is not None else None
    )
    cloned._scaling = rep._scaling.clone() if rep._scaling is not None else None
    cloned._rotation = rep._rotation.clone() if rep._rotation is not None else None
    cloned._opacity = rep._opacity.clone() if rep._opacity is not None else None
    return cloned


def _clone_mesh(rep: MeshExtractResult) -> MeshExtractResult:
    if rep.vertices is None or rep.faces is None:
        return rep
    vertices = rep.vertices.clone()
    faces = rep.faces.clone()
    vertex_attrs = rep.vertex_attrs.clone() if rep.vertex_attrs is not None else None
    new_mesh = MeshExtractResult(
        vertices=vertices, faces=faces, vertex_attrs=vertex_attrs, res=rep.res
    )
    if rep.tsdf_v is not None:
        new_mesh.tsdf_v = rep.tsdf_v.clone()
    if rep.tsdf_s is not None:
        new_mesh.tsdf_s = rep.tsdf_s.clone()
    if rep.reg_loss is not None:
        new_mesh.reg_loss = (
            rep.reg_loss.clone() if torch.is_tensor(rep.reg_loss) else rep.reg_loss
        )
    return new_mesh


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return axis / norm


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize_axis(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    trace = m00 + m11 + m22
    eps = 1e-8
    if trace > 0:
        s = torch.sqrt(trace + 1.0 + eps) * 2.0
        w = 0.25 * s
        inv_s = 1.0 / (s + eps)
        x = (m21 - m12) * inv_s
        y = (m02 - m20) * inv_s
        z = (m10 - m01) * inv_s
    elif m00 > m11 and m00 > m22:
        s = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0
        inv_s = 1.0 / (s + eps)
        w = (m21 - m12) * inv_s
        x = 0.25 * s
        y = (m01 + m10) * inv_s
        z = (m02 + m20) * inv_s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0
        inv_s = 1.0 / (s + eps)
        w = (m02 - m20) * inv_s
        x = (m01 + m10) * inv_s
        y = 0.25 * s
        z = (m12 + m21) * inv_s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0
        inv_s = 1.0 / (s + eps)
        w = (m10 - m01) * inv_s
        x = (m02 + m20) * inv_s
        y = (m12 + m21) * inv_s
        z = 0.25 * s
    quat = torch.stack([w, x, y, z])
    return torch.nn.functional.normalize(quat, dim=0)


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    quat = torch.stack([w, x, y, z], dim=-1)
    return torch.nn.functional.normalize(quat, dim=-1)


def _apply_transform_gaussian(rep: Gaussian, transform: np.ndarray) -> Gaussian:
    if rep._xyz is None:
        return _clone_gaussian(rep)
    xyz = rep.get_xyz
    device = xyz.device
    dtype = xyz.dtype
    transform_t = torch.as_tensor(transform, device=device, dtype=dtype)
    ones = torch.ones(xyz.shape[0], 1, device=device, dtype=dtype)
    homo = torch.cat([xyz, ones], dim=1)
    transformed_xyz = (homo @ transform_t.t())[:, :3]
    new_rep = _clone_gaussian(rep)
    new_rep.from_xyz(transformed_xyz)
    rotation_matrix = transform_t[:3, :3]
    if rep._rotation is not None:  # NOTE:: TODO check this.
        delta_quat = _matrix_to_quaternion(rotation_matrix)
        rest_quat = rep.get_rotation
        updated_quat = _quaternion_multiply(delta_quat[None, :], rest_quat)
        new_rep._rotation = updated_quat - new_rep.rots_bias[None, :]
    return new_rep


def _apply_transform_mesh(
    rep: MeshExtractResult, transform: np.ndarray
) -> MeshExtractResult:
    if rep.vertices is None or rep.faces is None:
        return _clone_mesh(rep)
    if rep.vertices.numel() == 0 or rep.faces.numel() == 0:
        return _clone_mesh(rep)

    vertices = rep.vertices
    device = vertices.device
    dtype = vertices.dtype
    transform_t = torch.as_tensor(transform, device=device, dtype=dtype)
    ones = torch.ones(vertices.shape[0], 1, device=device, dtype=dtype)
    homo = torch.cat([vertices, ones], dim=1)
    transformed_vertices = (homo @ transform_t.t())[:, :3]

    rotation_matrix = transform_t[:3, :3]
    vertex_attrs = None
    if rep.vertex_attrs is not None:
        vertex_attrs = rep.vertex_attrs.clone()
        if vertex_attrs.shape[-1] >= 6:
            normals = vertex_attrs[:, 3:6]
            rotated_normals = torch.matmul(normals, rotation_matrix.t())
            vertex_attrs[:, 3:6] = torch.nn.functional.normalize(
                rotated_normals, dim=-1
            )

    transformed_mesh = MeshExtractResult(
        vertices=transformed_vertices,
        faces=rep.faces.clone() if rep.faces is not None else rep.faces,
        vertex_attrs=vertex_attrs,
        res=rep.res,
    )

    if rep.tsdf_v is not None:
        tsdf_v = rep.tsdf_v
        ones_tsdf = torch.ones(tsdf_v.shape[0], 1, device=device, dtype=dtype)
        homo_tsdf = torch.cat([tsdf_v, ones_tsdf], dim=1)
        transformed_mesh.tsdf_v = (homo_tsdf @ transform_t.t())[:, :3]
    if rep.tsdf_s is not None:
        transformed_mesh.tsdf_s = rep.tsdf_s.clone()
    if rep.reg_loss is not None:
        transformed_mesh.reg_loss = (
            rep.reg_loss.clone() if torch.is_tensor(rep.reg_loss) else rep.reg_loss
        )

    return transformed_mesh


def _apply_transform(
    rep: Union[Gaussian, MeshExtractResult], transform: np.ndarray
) -> Union[Gaussian, MeshExtractResult]:
    if isinstance(rep, Gaussian):
        return _apply_transform_gaussian(rep, transform)
    if isinstance(rep, MeshExtractResult):
        return _apply_transform_mesh(rep, transform)
    raise TypeError(f"Unsupported representation type: {type(rep).__name__}")


def _build_hierarchy(nodes: List[Dict]) -> List[int]:
    lookup = {node.get("id", idx): node for idx, node in enumerate(nodes)}
    visited = set()
    order = []

    def dfs(node_id: int):
        if node_id in visited:
            return
        visited.add(node_id)
        node = lookup.get(node_id)
        if node is None:
            return
        parent = node.get("parent", -1)
        if parent not in (-1, None):
            dfs(parent)
        order.append(node_id)

    for key in lookup:
        dfs(key)
    return order


def _local_transform(node: Dict, t: float) -> np.ndarray:
    joint = node.get("joint", {}) or {}
    jtype = joint.get("type", "fixed") or "fixed"
    axis = joint.get("axis", {}) or {}
    axis_dir = _normalize_axis(axis.get("direction", [0.0, 0.0, 1.0]))
    axis_origin = np.asarray(axis.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
    joint_range = joint.get("range", [0.0, 0.0]) or [0.0, 0.0]
    if isinstance(joint_range, (int, float)):
        joint_range = [0.0, float(joint_range)]
    if len(joint_range) == 1:
        joint_range = [0.0, float(joint_range[0])]
    rmin, rmax = float(joint_range[0]), float(joint_range[1])
    transform = np.eye(4, dtype=np.float32)
    if jtype in ("revolute", "continuous"):
        angle = np.deg2rad(rmin + (rmax - rmin) * t)
        rot = _axis_angle_matrix(axis_dir, angle)
        transform[:3, :3] = rot
        transform[:3, 3] = axis_origin - rot @ axis_origin
    elif jtype in ("prismatic", "screw"):
        disp = rmin + (rmax - rmin) * t
        transform[:3, 3] = axis_dir * disp
    return transform


def animate_and_render(
    reps_parts: List[Gaussian],
    arti_info: List[dict],
    *,
    num_frames: int = 60,
    fps: int = 24,
    resolution: int = 512,
    bg_color=(1, 1, 1),
    cam_yaw: float = -0.785,
    cam_pitch: float = 0.2,
    cam_radius: float = 1.6,
    cam_fov: float = 60.0,
) -> Dict[str, np.ndarray]:
    if not reps_parts:
        raise ValueError("reps_parts is empty")
    if arti_info is None or len(arti_info) == 0:
        raise ValueError("arti_info is empty")

    nodes = list(arti_info)
    res_value = (
        int(resolution[0]) if isinstance(resolution, (list, tuple)) else int(resolution)
    )
    renderer = render_utils.get_renderer(
        reps_parts[0], resolution=res_value, bg_color=bg_color
    )
    renderer.rendering_options.resolution = res_value
    renderer.rendering_options.bg_color = tuple(float(c) for c in bg_color)
    cam_yaw, cam_pitch = np.pi, 0.5
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        cam_yaw, cam_pitch, cam_radius, cam_fov
    )

    node_assignment = nodes
    if len(node_assignment) != len(reps_parts):  ## NOTE:: may be error here ：TODO
        filtered = [node for node in nodes if node.get("name") != "base"]
        if len(filtered) == len(reps_parts):
            node_assignment = filtered
        else:
            node_assignment = nodes[: len(reps_parts)]

    node_to_rep = {}
    for rep, node in zip(reps_parts, node_assignment):
        node_id = node.get("id")
        if node_id is None:
            continue
        node_to_rep[node_id] = rep

    order = _build_hierarchy(nodes)
    id_to_node = {node.get("id"): node for node in nodes if node.get("id") is not None}
    identity = np.eye(4, dtype=np.float32)
    frames = []

    with torch.no_grad():
        for frame_idx in range(num_frames):
            t = frame_idx / max(num_frames - 1, 1)
            world_transforms: Dict[int, np.ndarray] = {-1: identity}
            for node_id in order:
                node = id_to_node.get(node_id)
                if node is None:
                    continue
                parent_id = node.get("parent", -1)
                parent_transform = world_transforms.get(parent_id, identity)
                local_transform = _local_transform(node, t)
                world_transforms[node_id] = parent_transform @ local_transform

            transformed_parts = []
            for node_id, base_rep in node_to_rep.items():
                transform = world_transforms.get(node_id, identity)
                transformed_parts.append(_apply_transform(base_rep, transform))

            if not transformed_parts:
                continue
            merged = merge_gaussians(transformed_parts)
            render_result = renderer.render(merged, extrinsics, intrinsics)
            frame = render_result["color"].detach().permute(1, 2, 0)
            frame = torch.clip(frame * 255.0, 0.0, 255.0).to(torch.uint8).cpu().numpy()
            frames.append(frame)

    if not frames:
        raise RuntimeError("Failed to generate animation frames")

    video = np.stack(frames, axis=0)
    return video


def animate(
    reps_parts: List[Union[Gaussian, MeshExtractResult]],
    arti_info: List[dict],
    t: float,
) -> Dict[str, np.ndarray]:
    if not reps_parts:
        raise ValueError("reps_parts is empty")
    if arti_info is None or len(arti_info) == 0:
        raise ValueError("arti_info is empty")

    nodes = list(arti_info)

    node_assignment = nodes
    if len(node_assignment) != len(reps_parts):  ## NOTE:: may be error here ：TODO
        filtered = [node for node in nodes if node.get("name") != "base"]
        if len(filtered) == len(reps_parts):
            node_assignment = filtered
        else:
            node_assignment = nodes[: len(reps_parts)]

    node_to_rep = {}
    for rep, node in zip(reps_parts, node_assignment):
        node_id = node.get("id")
        if node_id is None:
            continue
        node_to_rep[node_id] = rep

    order = _build_hierarchy(nodes)
    id_to_node = {node.get("id"): node for node in nodes if node.get("id") is not None}
    identity = np.eye(4, dtype=np.float32)
    frames = []
    with torch.no_grad():
        world_transforms: Dict[int, np.ndarray] = {-1: identity}
        for node_id in order:
            node = id_to_node.get(node_id)
            if node is None:
                continue
            parent_id = node.get("parent", -1)
            parent_transform = world_transforms.get(parent_id, identity)
            local_transform = _local_transform(node, t)
            world_transforms[node_id] = parent_transform @ local_transform

        transformed_parts = []
        for node_id, base_rep in node_to_rep.items():
            transform = world_transforms.get(node_id, identity)
            transformed_parts.append(_apply_transform(base_rep, transform))

        # merged = merge_gaussians(transformed_parts)

    return transformed_parts

    video = np.stack(frames, axis=0)
    return video
    import os
    import imageio

    output_dir = "./outputs/1arti_video"
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    gaussian_video_path = f"{output_dir}/object__gs.mp4"
    imageio.mimsave(gaussian_video_path, video, fps=20)
    print(f"Save video to {gaussian_video_path}")
    for i, video in enumerate(videos):
        output_dir = os.path.join(self.output_dir, "samples", suffix, f"videos_{key}")
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)
        gaussian_video_path = f"{output_dir}/object_{i}_gs.mp4"
        imageio.mimsave(gaussian_video_path, video, fps=20)


# animate_and_render = manipulate_and_render


def rescale_axis(jtype, axis_d, axis_o, box_center):
    """
    Function to rescale the axis for rendering

    Args:
    - jtype (int): joint type
    - axis_d (np.array): axis direction
    - axis_o (np.array): axis origin
    - box_center (np.array): bounding box center

    Returns:
    - center (np.array): rescaled axis origin
    - axis_d (np.array): rescaled axis direction
    """
    if jtype == 0 or jtype == 1:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if jtype == 3 or jtype == 4:
        center = box_center
    else:
        ## NOTE: to make the axis align with x/y/z axes
        is_trick_dir = True
        if is_trick_dir and jtype == 2:
            indicator = np.sign(axis_d)
            norm = np.linalg.norm(axis_d)
            if norm < 1e-8:
                axis_d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                indicator = np.sign(axis_d)
            else:
                axis_d = axis_d / norm
            idx = int(np.argmax(np.abs(axis_d)))
            sign_val = float(indicator[idx]) if indicator[idx] != 0 else 1.0
            new_axis = np.zeros_like(axis_d)
            new_axis[idx] = sign_val
            axis_d = new_axis
            center = axis_o
        else:
            center = axis_o + np.dot(axis_d, box_center - axis_o) * axis_d
    return center.tolist(), axis_d.tolist()


def convert_data_range(x):
    """postprocessing: convert the raw model output to the original range, following CAGE"""
    # x = x.reshape(-1, 24)  # (K, 30)
    # aabb_max = x[:, 0:3]
    # aabb_min = x[:, 3:6]
    # center = (aabb_max + aabb_min) / 2.0
    # size = (aabb_max - aabb_min).clip(min=5e-3)

    j_type = np.mean(x[:, 0:6], axis=1)
    j_type = ((j_type + 0.5) * 5).clip(min=1.0, max=5.0).round()

    axis_d = x[:, 6:9]
    axis_d = axis_d / (
        np.linalg.norm(axis_d, axis=1, keepdims=True) + np.finfo(float).eps
    )
    axis_o = x[:, 9:12]

    j_range = (x[:, 12:14] + x[:, 14:16] + x[:, 16:18]) / 3
    j_range = j_range.clip(min=-1.0, max=1.0)
    j_range[:, 0] = j_range[:, 0] * 360
    j_range[:, 1] = j_range[:, 1]

    label = np.mean(x[:, 18:24], axis=1)
    label = ((label + 0.8) * 5).clip(min=0.0, max=7.0).round()
    return {
        # "center": center,
        # "size": size,
        "type": j_type,
        "axis_d": axis_d,
        "axis_o": axis_o,
        "range": j_range,
        "label": label,
    }


def parse_tree(data, n_nodes, par=None, adj=None):
    tree = []
    # convert to json format
    for i in range(n_nodes):
        node = {"id": i}
        try:
            node["name"] = sem_ref["bwd"][int(data["label"][i].item())]
        except Exception as e:
            node["name"] = "unknown"
            pass
        if par is not None and adj is not None:
            node["parent"] = int(par[i])
            node["children"] = [
                int(child) for child in np.where(adj[i] == 1)[0] if child != par[i]
            ]
        node["aabb"] = {}
        node["aabb"]["center"] = (
            data["center"][i].tolist() if "center" in data else None
        )
        node["aabb"]["size"] = data["size"][i].tolist() if "size" in data else None
        node["joint"] = {}
        if node["name"] == "base":
            node["joint"]["type"] = "fixed"
        else:
            node["joint"]["type"] = joint_ref["bwd"][int(data["type"][i].item())]
        if node["joint"]["type"] == "fixed":
            node["joint"]["range"] = [0.0, 0.0]
        elif node["joint"]["type"] == "revolute":
            node["joint"]["range"] = [0.0, float(data["range"][i][0])]
        elif node["joint"]["type"] == "continuous":
            node["joint"]["range"] = [0.0, 360.0]
        elif node["joint"]["type"] == "prismatic" or node["joint"]["type"] == "screw":
            node["joint"]["range"] = [0.0, float(data["range"][i][1])]
        node["joint"]["axis"] = {}
        # relocate the axis to visualize well
        axis_o, axis_d = rescale_axis(
            int(data["type"][i].item()),
            data["axis_d"][i],
            data["axis_o"][i],
            data["center"][i] if "center" in data else np.array([0.0, 0.0, 0.0]),
        )
        node["joint"]["axis"]["direction"] = axis_d
        node["joint"]["axis"]["origin"] = axis_o
        # append node to the tree
        tree.append(node)
    return tree


def convert_data_2_info(x):
    # convert the data to original range
    data = convert_data_range(x)

    n_nodes = x.shape[0]
    # parse the tree
    tree = parse_tree(
        data,
        n_nodes,
    )
    return tree


def convert_json(x, c, prefix=""):
    out = {"meta": {}, "diffuse_tree": []}
    n_nodes = c[f"{prefix}n_nodes"][0].item()
    par = c[f"{prefix}parents"][0].cpu().numpy().tolist()
    adj = c[f"{prefix}adj"][0].cpu().numpy()
    np.fill_diagonal(adj, 0)  # remove self-loop for the root node
    if f"{prefix}obj_cat" in c:
        out["meta"]["obj_cat"] = c[f"{prefix}obj_cat"][0]

    # convert the data to original range
    data = convert_data_range(x)
    # parse the tree
    tree = parse_tree(data, n_nodes, par, adj)
    out["diffuse_tree"] = tree
    return out
