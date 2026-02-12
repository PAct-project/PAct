#!/usr/bin/env python3
"""Utility to convert OmniPart articulation JSON files into URDF descriptions."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

Vector3 = Tuple[float, float, float]


@dataclass
class ConverterConfig:
    """Runtime configuration derived from CLI arguments."""

    json_path: Path
    output_path: Path
    asset_root: Path
    mesh_priority: str
    geometry_format: Optional[str]
    absolute_mesh_paths: bool
    robot_name: Optional[str]
    min_mass: float
    limit_effort: float
    limit_velocity: float


@dataclass
class NodeWrapper:
    """Lightweight helper that carries precomputed metadata for each node."""

    node: Mapping
    name: str
    origin: Vector3


def parse_args() -> ConverterConfig:
    parser = argparse.ArgumentParser(
        description="Convert an OmniPart articulation JSON file into a URDF." ,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input object_merge_fixed.json file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Target path for the generated URDF file.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Base directory used to resolve mesh paths (defaults to the JSON parent).",
    )
    parser.add_argument(
        "--mesh-priority",
        choices=("obj", "ply"),
        default="obj",
        help="Which mesh list to prioritize when both OBJ and PLY entries exist.",
    )
    geometry_group = parser.add_mutually_exclusive_group()
    geometry_group.add_argument(
        "--glb",
        action="store_true",
        help="Use GLB assets for visual/collision geometry when available.",
    )
    geometry_group.add_argument(
        "--ply",
        action="store_true",
        help="Use PLY assets for visual/collision geometry exclusively.",
    )
    geometry_group.add_argument(
        "--obj",
        action="store_true",
        help="Use OBJ assets for visual/collision geometry exclusively.",
    )
    parser.add_argument(
        "--absolute-mesh-paths",
        action="store_true",
        help="Emit absolute mesh paths instead of making them relative to the URDF file.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="Override the robot name stored inside the URDF.",
    )
    parser.add_argument(
        "--min-mass",
        type=float,
        default=1.0,
        help="Minimum mass (kg) assigned to any link.",
    )
    parser.add_argument(
        "--limit-effort",
        type=float,
        default=50.0,
        help="Default joint effort limit used when the JSON file does not specify one.",
    )
    parser.add_argument(
        "--limit-velocity",
        type=float,
        default=1.0,
        help="Default joint velocity limit used when the JSON file does not specify one.",
    )
    args = parser.parse_args()

    json_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    asset_root = (
        args.asset_root.expanduser().resolve() if args.asset_root else json_path.parent
    )
    geometry_format = "glb" if args.glb else ("ply" if args.ply else None)

    return ConverterConfig(
        json_path=json_path,
        output_path=output_path,
        asset_root=asset_root,
        mesh_priority=args.mesh_priority,
        geometry_format=geometry_format,
        absolute_mesh_paths=args.absolute_mesh_paths,
        robot_name=args.robot_name,
        min_mass=max(1e-5, args.min_mass),
        limit_effort=max(1e-6, args.limit_effort),
        limit_velocity=max(1e-6, args.limit_velocity),
    )


def sanitize_name(raw: Optional[str], suffix: str) -> str:
    base = (raw or "link").strip()
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", base)
    if not base:
        base = "link"
    if base[0].isdigit():
        base = f"l_{base}"
    return f"{base}_{suffix}" if suffix else base


def vector_from(data: Optional[Sequence], fallback: Vector3 = (0.0, 0.0, 0.0)) -> Vector3:
    if not data:
        return fallback
    values = list(data)[:3]
    while len(values) < 3:
        values.append(0.0)
    return float(values[0]), float(values[1]), float(values[2])


def vector_sub(lhs: Vector3, rhs: Vector3) -> Vector3:
    return lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]


def vector_neg(vec: Vector3) -> Vector3:
    return (-vec[0], -vec[1], -vec[2])


def format_xyz(vec: Vector3) -> str:
    return " ".join(f"{value:.6f}" for value in vec)


def normalize(vec: Vector3, fallback: Vector3 = (0.0, 0.0, 1.0)) -> Vector3:
    length = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if length < 1e-8:
        return fallback
    return vec[0] / length, vec[1] / length, vec[2] / length


def compute_mass_and_inertia(size: Vector3, min_mass: float) -> Tuple[float, Vector3]:
    sx, sy, sz = max(size[0], 1e-4), max(size[1], 1e-4), max(size[2], 1e-4)
    approx_mass = max(min_mass, abs(sx * sy * sz))
    coeff = approx_mass / 12.0
    ixx = coeff * (sy ** 2 + sz ** 2)
    iyy = coeff * (sx ** 2 + sz ** 2)
    izz = coeff * (sx ** 2 + sy ** 2)
    return approx_mass, (ixx, iyy, izz)


def choose_mesh_list(node: Mapping, config: ConverterConfig) -> Iterable[str]:
    if config.geometry_format == "glb":
        return node.get("glb") or []
    if config.geometry_format == "ply":
        return node.get("plys") or []
    if config.geometry_format == "obj":
        return node.get("objs") or []
    objs = node.get("objs") or []
    plys = node.get("plys") or []
    glbs = node.get("glb") or []
    return objs + plys + glbs if config.mesh_priority == "obj" else plys + objs + glbs


def materialize_mesh_paths(
    rel_paths: Iterable[str],
    asset_root: Path,
    output_dir: Path,
    absolute: bool,
) -> List[str]:
    urdf_paths: List[str] = []
    for rel_path in rel_paths:
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        abs_path = (asset_root / rel_path).resolve()
        if absolute:
            urdf_paths.append(abs_path.as_posix())
        else:
            try:
                relative = os.path.relpath(abs_path, output_dir)
                urdf_paths.append(Path(relative).as_posix())
            except ValueError:
                urdf_paths.append(abs_path.as_posix())
    return urdf_paths


def determine_link_origins(nodes: Mapping[int, Mapping]) -> Dict[int, Vector3]:
    origins: Dict[int, Vector3] = {}
    for node_id, node in nodes.items():
        parent_id = node.get("parent", -1)
        if parent_id == -1:
            origins[node_id] = (0.0, 0.0, 0.0)
            continue
        joint = node.get("joint") or {}
        axis = joint.get("axis") or {}
        axis_origin = vector_from(axis.get("origin"))
        if joint.get("type") != "fixed":
            origins[node_id] = axis_origin
        else:
            aabb = node.get("aabb") or {}
            origins[node_id] = vector_from(aabb.get("center"), axis_origin)
    return origins


def ensure_single_root(nodes: MutableMapping[int, Mapping]) -> int:
    roots = [node_id for node_id, node in nodes.items() if node.get("parent", -1) == -1]
    if not roots:
        raise ValueError("No root nodes detected in the JSON file.")
    if len(roots) == 1:
        return roots[0]
    next_id = max(nodes) + 1
    nodes[next_id] = {
        "id": next_id,
        "name": "virtual_root",
        "parent": -1,
        "children": roots,
        "joint": {"type": "fixed", "range": [0.0, 0.0], "axis": {"origin": [0, 0, 0], "direction": [0, 0, 1]}},
        "objs": [],
        "plys": [],
        "aabb": {"center": [0, 0, 0], "size": [0, 0, 0]},
    }
    for root_id in roots:
        nodes[root_id]["parent"] = next_id
    return next_id


def build_node_wrappers(nodes: Mapping[int, Mapping]) -> Dict[int, NodeWrapper]:
    origins = determine_link_origins(nodes)
    taken_names: set[str] = set()
    wrappers: Dict[int, NodeWrapper] = {}
    for node_id in sorted(nodes):
        node = nodes[node_id]
        sanitized = sanitize_name(node.get("name"), str(node_id))
        if sanitized in taken_names:
            suffix = 1
            candidate = f"{sanitized}_{suffix}"
            while candidate in taken_names:
                suffix += 1
                candidate = f"{sanitized}_{suffix}"
            sanitized = candidate
        taken_names.add(sanitized)
        wrappers[node_id] = NodeWrapper(node=node, name=sanitized, origin=origins.get(node_id, (0.0, 0.0, 0.0)))
    return wrappers


def build_robot_element(data: Mapping, config: ConverterConfig) -> Tuple[Element, Dict[int, NodeWrapper]]:
    node_map: Dict[int, Mapping] = {entry["id"]: entry for entry in data.get("diffuse_tree", [])}
    if not node_map:
        raise ValueError("The input JSON does not contain any nodes under 'diffuse_tree'.")

    root_id = ensure_single_root(node_map)
    wrappers = build_node_wrappers(node_map)

    robot_name = config.robot_name or sanitize_name(data.get("meta", {}).get("obj_cat"), "robot")
    robot = Element("robot", name=robot_name)

    output_dir = config.output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for node_id, wrapper in wrappers.items():
        node = wrapper.node
        link_elem = SubElement(robot, "link", name=wrapper.name)
        aabb = node.get("aabb") or {}
        size = vector_from(aabb.get("size"), (0.01, 0.01, 0.01))
        center = vector_from(aabb.get("center"))
        mass, inertia = compute_mass_and_inertia(size, config.min_mass)
        inertial = SubElement(link_elem, "inertial")
        SubElement(inertial, "origin", xyz=format_xyz(vector_sub(center, wrapper.origin)), rpy="0 0 0")
        SubElement(inertial, "mass", value=f"{mass:.6f}")
        SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia[0]:.6f}",
            iyy=f"{inertia[1]:.6f}",
            izz=f"{inertia[2]:.6f}",
            ixy="0.0",
            ixz="0.0",
            iyz="0.0",
        )

        mesh_candidates = list(choose_mesh_list(node, config))
        mesh_paths = materialize_mesh_paths(
            mesh_candidates,
            config.asset_root,
            output_dir,
            config.absolute_mesh_paths,
        )

        if mesh_paths:
            for mesh_path in mesh_paths:
                visual = SubElement(link_elem, "visual")
                SubElement(visual, "origin", xyz=format_xyz(vector_neg(wrapper.origin)), rpy="0 0 0")
                geometry = SubElement(visual, "geometry")
                SubElement(geometry, "mesh", filename=mesh_path)
                collision = SubElement(link_elem, "collision")
                SubElement(collision, "origin", xyz=format_xyz(vector_neg(wrapper.origin)), rpy="0 0 0")
                col_geometry = SubElement(collision, "geometry")
                SubElement(col_geometry, "mesh", filename=mesh_path)
        else:
            # Fallback to boxes when no explicit meshes exist.
            visual = SubElement(link_elem, "visual")
            SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
            geometry = SubElement(visual, "geometry")
            SubElement(geometry, "box", size=format_xyz(size))
            collision = SubElement(link_elem, "collision")
            SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            col_geometry = SubElement(collision, "geometry")
            SubElement(col_geometry, "box", size=format_xyz(size))

    # Build joints after links exist.
    for node_id, wrapper in wrappers.items():
        node = wrapper.node
        parent_id = node.get("parent", -1)
        if parent_id == -1:
            continue
        parent_wrapper = wrappers[parent_id]
        joint_info = node.get("joint") or {}
        joint_type = joint_info.get("type", "fixed").lower()
        joint_name = sanitize_name(node.get("name"), f"joint_{node_id}")
        joint_elem = SubElement(robot, "joint", name=joint_name, type=joint_type)
        SubElement(joint_elem, "parent", link=parent_wrapper.name)
        SubElement(joint_elem, "child", link=wrapper.name)
        joint_origin = vector_sub(wrapper.origin, parent_wrapper.origin)
        SubElement(joint_elem, "origin", xyz=format_xyz(joint_origin), rpy="0 0 0")
        axis_direction = vector_from((joint_info.get("axis") or {}).get("direction"))
        if joint_type != "fixed":
            SubElement(joint_elem, "axis", xyz=format_xyz(normalize(axis_direction)))
            joint_range = joint_info.get("range") or []
            if joint_type in {"revolute", "prismatic"} and len(joint_range) >= 2:
                lower, upper = float(joint_range[0]), float(joint_range[1])
                if joint_type == "revolute":
                    lower, upper = math.radians(lower), math.radians(upper)
                if lower > upper:
                    lower, upper = upper, lower
                SubElement(
                    joint_elem,
                    "limit",
                    lower=f"{lower:.6f}",
                    upper=f"{upper:.6f}",
                    effort=f"{config.limit_effort:.6f}",
                    velocity=f"{config.limit_velocity:.6f}",
                )
        else:
            SubElement(joint_elem, "axis", xyz="0 0 0")

    return robot, wrappers


def prettify(element: Element) -> str:
    rough = tostring(element, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def main() -> None:
    config = parse_args()
    with config.json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    robot_element, _ = build_robot_element(data, config)
    pretty_xml = prettify(robot_element)
    config.output_path.write_text(pretty_xml, encoding="utf-8")
    print(f"URDF written to {config.output_path}")


if __name__ == "__main__":
    main()
