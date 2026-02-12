#!/usr/bin/env python3
"""
Animate an articulated object by sweeping each joint from its min to max range
and render a video.

Usage:
    python scripts/animate_articulated.py /path/to/object.json --out outputs/anim.mp4

Requires: trimesh, numpy, imageio
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

import imageio

# pyrender is optional; used for faster offscreen GL rendering
try:
    import pyrender

    _HAS_PYRENDER = True
except Exception:
    pyrender = None
    _HAS_PYRENDER = False
import pyglet


import pyglet

window = pyglet.window.Window(visible=False)

import pyglet.gl as gl

print("OpenGL version:", gl.gl_info.get_version())
print("OpenGL renderer:", gl.gl_info.get_renderer())

window.close()
print(pyglet.gl.gl_info.get_version())
print(pyglet.gl.gl_info.get_renderer())


def load_mesh_for_node(node: dict, base_dir: str) -> trimesh.Trimesh:
    # prefer plys then objs; load and concatenate all files for the node
    candidates = []
    if node.get("plys"):
        candidates = list(node.get("plys"))
    elif node.get("objs"):
        candidates = list(node.get("objs"))

    meshes = []
    for rel in candidates:
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.exists(path):
            alt = os.path.join(base_dir, rel.lstrip("./"))
            if os.path.exists(alt):
                path = alt
            else:
                continue
        try:
            m = trimesh.load(path, force="mesh")
            if isinstance(m, trimesh.Scene):
                if len(m.geometry) == 0:
                    continue
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            if isinstance(m, (list, tuple)):
                m = trimesh.util.concatenate(tuple(m))
            meshes.append(m)
        except Exception:
            continue
    print("meshes loaded for node:", len(meshes), candidates, node)
    if len(meshes) == 0:
        return None
    if len(meshes) == 1:
        return meshes[0]

    return trimesh.util.concatenate(tuple(meshes))


def build_hierarchy(nodes):
    # return list of node indices in parent-first order
    lookup = {n["id"]: n for n in nodes}
    order = []
    visited = set()

    def dfs(nid):
        if nid in visited:
            return
        visited.add(nid)
        n = lookup[nid]
        p = n.get("parent")
        if p is not None and p != -1:
            dfs(p)
        order.append(nid)

    for n in nodes:
        dfs(n["id"])
    return order


def color_from_id(i: int):
    rng = np.random.RandomState(i + 1234)
    return (rng.rand(3) * 200 + 55).astype(np.uint8)


def animate(
    object_json,
    out_path,
    frames=60,
    resolution=(1024, 768),
    fps=24,
    offscreen=False,
    cam_az=-0.785,
    cam_el=-0.2,
    cam_distance=None,
):
    base_dir = os.path.dirname(object_json)
    with open(object_json, "r") as f:
        data = json.load(f)
    nodes = data.get("diffuse_tree", [])
    id_to_node = {n["id"]: n for n in nodes}

    # load meshes and align to aabb.center if present
    meshes = {}
    for n in nodes:
        m = load_mesh_for_node(n, base_dir)
        if m is None:
            continue
        aabb = n.get("aabb")
        if aabb and "center" in aabb:
            target = np.array(aabb["center"], dtype=float)
            cur = m.bounds.mean(axis=0)
            trans = target - cur
            if np.linalg.norm(trans) > 1e-9:
                m = m.copy()
                m.apply_translation(trans)
        meshes[n["id"]] = m
    print("Loaded meshes for parts:", list(meshes.keys()))
    # assign per-part colors and transparency to match visualize_articulated
    part_colors = {}
    part_alpha = 160
    print(nodes)
    for n in nodes:
        nid = n["id"]
        mesh = meshes.get(nid)
        if mesh is None:
            continue
        col = color_from_id(nid).astype(np.uint8)
        # detect texture
        has_texture = False
        try:
            has_texture = getattr(mesh.visual, "kind", None) == "texture"
            if not has_texture:
                uv = getattr(mesh.visual, "uv", None)
                if uv is not None and len(uv) > 0:
                    has_texture = True
        except Exception:
            has_texture = False
        print("has_texture", has_texture)
        # if True :
        if not has_texture:
            nvert = len(mesh.vertices)
            rgba = np.hstack(
                [
                    np.tile(col.reshape(1, 3), (nvert, 1)),
                    np.full((nvert, 1), part_alpha, dtype=np.uint8),
                ]
            )
            print("color", rgba)
            try:
                mesh.visual.vertex_colors = rgba
            except Exception:
                pass
            part_colors[nid] = np.array(
                [int(col[0]), int(col[1]), int(col[2]), int(part_alpha)], dtype=np.uint8
            )
        else:
            # try set material baseColorFactor alpha and alphaMode
            try:
                mat = getattr(mesh.visual, "material", None)
                a = float(part_alpha) / 255.0
                if mat is not None:
                    try:
                        mat.baseColorFactor = [
                            float(col[0]) / 255.0,
                            float(col[1]) / 255.0,
                            float(col[2]) / 255.0,
                            a,
                        ]
                    except Exception:
                        try:
                            setattr(
                                mat,
                                "baseColorFactor",
                                [
                                    float(col[0]) / 255.0,
                                    float(col[1]) / 255.0,
                                    float(col[2]) / 255.0,
                                    a,
                                ],
                            )
                        except Exception:
                            pass
                    try:
                        mat.alphaMode = "BLEND"
                    except Exception:
                        try:
                            setattr(mat, "alphaMode", "BLEND")
                        except Exception:
                            pass
                part_colors[nid] = np.array(
                    [int(col[0]), int(col[1]), int(col[2]), int(part_alpha)],
                    dtype=np.uint8,
                )
            except Exception:
                part_colors[nid] = np.array(
                    [int(col[0]), int(col[1]), int(col[2]), int(part_alpha)],
                    dtype=np.uint8,
                )

    order = build_hierarchy(nodes)

    # prepare ranges and axis info
    joint_info = {}
    for n in nodes:
        jid = n["id"]
        j = n.get("joint", {})
        jtype = j.get("type", "fixed")
        if jtype == "revolute":
            rmin, rmax = j.get("range", [0, 0])
            axis_o = np.array(j.get("axis", {}).get("origin", [0, 0, 0]), dtype=float)
            axis_d = np.array(
                j.get("axis", {}).get("direction", [0, 0, 1]), dtype=float
            )
            if np.linalg.norm(axis_d) == 0:
                axis_d = np.array([0, 0, 1], dtype=float)
            axis_d = axis_d / np.linalg.norm(axis_d)
            joint_info[jid] = ("revolute", axis_o, axis_d, float(rmin), float(rmax))
        elif jtype == "prismatic":
            rmin, rmax = j.get("range", [0, 0])
            axis_o = np.array(j.get("axis", {}).get("origin", [0, 0, 0]), dtype=float)
            axis_d = np.array(
                j.get("axis", {}).get("direction", [0, 0, 1]), dtype=float
            )
            if np.linalg.norm(axis_d) == 0:
                axis_d = np.array([0, 0, 1], dtype=float)
            axis_d = axis_d / np.linalg.norm(axis_d)
            joint_info[jid] = ("prismatic", axis_o, axis_d, float(rmin), float(rmax))
        else:
            joint_info[jid] = ("fixed", None, None, 0.0, 0.0)

    # prepare camera: center on whole model
    all_mesh = (
        trimesh.util.concatenate(tuple([m for m in meshes.values()]))
        if len(meshes) > 0
        else None
    )
    center = all_mesh.bounds.mean(axis=0) if all_mesh is not None else np.zeros(3)
    max_dim = max(all_mesh.extents) if all_mesh is not None else 1.0
    distance = max_dim * 2.0 if cam_distance is None else float(cam_distance)

    writer = imageio.get_writer(out_path, fps=fps)

    # prepare an offscreen renderer if requested and available
    renderer = None
    if offscreen:
        if not _HAS_PYRENDER:
            print("pyrender not available; falling back to trimesh save_image")
            offscreen = False
        else:
            try:
                renderer = pyrender.OffscreenRenderer(
                    viewport_width=resolution[0], viewport_height=resolution[1]
                )
            except Exception as e:
                print("Failed to create OffscreenRenderer:", e)
                renderer = None
                offscreen = False

    # default camera angles used by trimesh previously
    default_az = -0.785
    default_el = -0.2

    for f in range(frames):
        t = f / (frames - 1) if frames > 1 else 1.0
        # compute transforms
        world_T = {}
        world_T[-1] = np.eye(4)
        for nid in order:
            node = id_to_node[nid]
            p = node.get("parent", -1)
            parent_T = world_T.get(p, np.eye(4))
            print(parent_T)
            ### if all movable part then thhe parent_T is identity
            info = joint_info.get(nid, ("fixed", None, None, 0.0, 0.0))
            if info[0] == "revolute":
                _, origin, axis, rmin, rmax = info
                angle_deg = rmin + (rmax - rmin) * t
                angle = np.deg2rad(angle_deg)
                R = rotation_matrix(angle, axis, origin)
                local_T = R
            elif info[0] == "prismatic":
                _, origin, axis, rmin, rmax = info
                disp = rmin + (rmax - rmin) * t
                # translate along axis by disp from origin
                # translation matrix computed without origin offset because translation along axis in world
                local_T = translation_matrix(axis * disp)
            else:
                local_T = np.eye(4)
            world_T[nid] = parent_T.dot(local_T)

        # build scene for this frame and render
        if offscreen and renderer is not None:
            # use pyrender offscreen: create a pyrender.Scene and add meshes (already transformed)
            pscene = pyrender.Scene(
                bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3]
            )
            for nid, mesh in meshes.items():
                m2 = mesh.copy()
                T = world_T.get(nid, np.eye(4))
                m2.apply_transform(T)
                try:
                    pm = pyrender.Mesh.from_trimesh(m2, smooth=False)
                    pscene.add(pm)
                except Exception as e:
                    # fallback: skip this mesh
                    print("pyrender conversion failed for part", nid, e)
            # camera: create pyrender camera
            # try to match trimesh set_camera angles: azimuth, elevation
            az, el = cam_az, cam_el
            # vertical fov to match trimesh (use second value if available)
            try:
                yfov = np.deg2rad(45.0)
                cam = pyrender.PerspectiveCamera(yfov=yfov)
            except Exception:
                cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))

            # compute camera position in spherical coordinates around center
            x = np.cos(el) * np.cos(az)
            y = np.cos(el) * np.sin(az)
            z = np.sin(el)
            cam_pos = center + distance * np.array([x, y, z])

            # build look-at rotation
            forward = (center - cam_pos).astype(float)
            fn = np.linalg.norm(forward)
            if fn < 1e-8:
                forward = np.array([0.0, 0.0, -1.0])
            else:
                forward = forward / fn
            up_world = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up_world)
            rn = np.linalg.norm(right)
            if rn < 1e-8:
                up_world = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, up_world)
                rn = np.linalg.norm(right)
            right = right / max(rn, 1e-8)
            up_cam = np.cross(right, forward)

            R = np.column_stack((right, up_cam, -forward))
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = R
            cam_pose[:3, 3] = cam_pos
            try:
                pscene.add(cam, pose=cam_pose)
                pass
            except Exception:
                pass
            try:
                color, depth = renderer.render(pscene)
                writer.append_data(color)
            except Exception as e:
                print("pyrender render failed at frame", f, e)
                # fallback to trimesh save_image
                scene = trimesh.Scene()
                for nid, mesh in meshes.items():
                    m2 = mesh.copy()
                    T = world_T.get(nid, np.eye(4))
                    m2.apply_transform(T)
                    scene.add_geometry(m2, f"part_{nid}")
                img = scene.save_image(resolution=resolution, visible=True)
                if img is None:
                    print("frame render failed at", f)
                    continue
                writer.append_data(imageio.imread(img))
        else:
            scene = trimesh.Scene()
            for nid, mesh in meshes.items():
                m2 = mesh.copy()
                T = world_T.get(nid, np.eye(4))
                m2.apply_transform(T)
                scene.add_geometry(m2, f"part_{nid}")
            try:
                scene.set_camera(
                    angles=(-0.785, 0.2, 0),
                    distance=distance,
                    center=center,
                    fov=(60, 45),
                )
            except Exception:
                pass
            img = scene.save_image(resolution=resolution, visible=False)
            # img = scene.save_image(resolution=resolution, visible=True)
            if img is None:
                print("frame render failed at", f)
                continue
            writer.append_data(imageio.imread(img))
        print(f"Wrote frame {f+1}/{frames}")

    writer.close()
    print("Saved animation to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("object_json")
    parser.add_argument("--out", default="outputs/animation.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--res", type=str, default="1024x768")
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="use pyrender OffscreenRenderer if available",
    )
    parser.add_argument(
        "--cam-az", type=float, default=-0.785, help="camera azimuth (radians)"
    )
    parser.add_argument(
        "--cam-el", type=float, default=-0.2, help="camera elevation (radians)"
    )
    parser.add_argument(
        "--cam-distance", type=float, default=None, help="camera distance from center"
    )
    args = parser.parse_args()
    w, h = map(int, args.res.split("x"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    animate(
        args.object_json,
        args.out,
        frames=args.frames,
        resolution=(w, h),
        fps=args.fps,
        offscreen=args.offscreen,
        cam_az=args.cam_az,
        cam_el=args.cam_el,
        cam_distance=args.cam_distance,
    )
# python3 scripts/animate_articulated.py Datasets/example_datasets/40147/object.json \
#   --out outputs/40147_animation_camtest.mp4 --frames 4 --fps 4 --res 640x480 --offscreen \
#   --cam-az -0.785 --cam-el -0.2
