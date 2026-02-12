import gradio as gr
import spaces
import os
import json
import zipfile
from pathlib import Path
import numpy as np
import trimesh
import time
import traceback
import torch
from PIL import Image
import cv2
import shutil
from glob import glob
import copy
import imageio.v3 as iio
from torch.utils.data import DataLoader
from segment_anything import SamAutomaticMaskGenerator, build_sam
import imageio

from modules.pact import datasets
from modules.pact.pipelines import PActPipeline, RenderConfig, ExportConfig
from modules.label_2d_mask.visualizer import Visualizer
from transformers import AutoModelForImageSegmentation
from scripts.json_to_urdf import ConverterConfig, build_robot_element, prettify

from modules.label_2d_mask.label_parts import (
    prepare_image,
    get_sam_mask,
    get_mask,
    clean_segment_edges,
    resize_and_pad_to_square,
    size_th as DEFAULT_SIZE_TH,
)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MAX_SEED = np.iinfo(np.int32).max
TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_ROOT, exist_ok=True)

sam_mask_generator = None
rmbg_model = None
pact_pipeline = None

size_th = DEFAULT_SIZE_TH

TAG_DIR = ""


def prepare_models(sam_ckpt_path, pipeline_path: str = None):
    global sam_mask_generator, rmbg_model, pact_pipeline
    if sam_ckpt_path is None:
        raise ValueError("sam_ckpt_path is required")

    if sam_mask_generator is None:
        print("Loading SAM model...")
        sam_model = build_sam(checkpoint=sam_ckpt_path).to(device=DEVICE)
        sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

    if rmbg_model is None:
        print("Loading BriaRMBG 2.0 model...")
        rmbg_model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        rmbg_model.to(DEVICE)
        rmbg_model.eval()

    if pact_pipeline is None:
        resolved_pipeline_path = pipeline_path
        print(f"Loading PAct pipeline from {resolved_pipeline_path}...")
        pact_pipeline = PActPipeline.from_pretrained(
            resolved_pipeline_path, revision="main"
        )
        pact_pipeline.to(DEVICE)
        pact_pipeline.verbose = False

    print("Models ready")
    return sam_mask_generator, rmbg_model, pact_pipeline


def _colorize_labels(label_map: np.ndarray) -> np.ndarray:
    """Create a simple color visualization for a label map."""
    label_map = label_map.astype(np.int32)
    vis = np.ones((*label_map.shape, 3), dtype=np.uint8) * 255
    unique_ids = [i for i in np.unique(label_map) if i >= 0]
    for i, uid in enumerate(unique_ids):
        color = np.array(
            [
                (i * 50 + 80) % 256,
                (i * 120 + 40) % 256,
                (i * 180 + 20) % 256,
            ],
            dtype=np.uint8,
        )
        vis[label_map == uid] = color
    vis[label_map == 0] = np.array(
        [255, 255, 255], dtype=np.uint8
    )  # background as white

    return vis


@spaces.GPU
def process_image(image_path, mask_path, req: gr.Request):
    """Process image and generate initial segmentation. If mask_path is provided, skip SAM."""
    global size_th

    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    global TAG_DIR
    TAG_DIR = os.path.basename(image_path).split(".")[0].replace(" ", "_")
    user_dir = os.path.join(user_dir, TAG_DIR)
    os.makedirs(user_dir, exist_ok=True)

    img_name = os.path.basename(image_path).split(".")[0]

    def _load_mask_to_labels(path: str) -> np.ndarray:
        mask = iio.imread(path)
        if mask.ndim == 2:
            labels = mask.astype(np.int32)
        else:
            mask = mask[..., :3]
            h, w, c = mask.shape
            flat = mask.reshape(-1, c)
            uniq, inv = np.unique(flat, axis=0, return_inverse=True)
            labels = inv.reshape(h, w).astype(np.int32)
        return labels

    # If provided mask exists, bypass SAM and use it directly
    if mask_path and os.path.exists(mask_path):

        rgba_path = os.path.join(user_dir, f"{img_name}_processed.png")
        shutil.copy(image_path, rgba_path)

        img = Image.open(image_path).convert("RGB")
        processed_image = prepare_image(img, rmbg_net=rmbg_model.to(DEVICE))
        white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
        white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
        image = np.array(white_bg_img.convert("RGB"))
        rgba_path = os.path.join(user_dir, f"{img_name}_processed.png")
        processed_image.resize(img.size, resample=Image.LANCZOS).save(rgba_path)

        labels = _load_mask_to_labels(mask_path)
        group_ids = labels.astype(np.int32)  # start at 0

        mask_to_save = (group_ids).astype(np.float32)
        mask_to_save = mask_to_save.reshape(*mask_to_save.shape, 1).repeat(3, axis=-1)
        save_mask_path = os.path.join(user_dir, f"{img_name}_mask.exr")
        # cv2.imwrite(save_mask_path, mask_to_save)
        iio.imwrite(save_mask_path, mask_to_save)

        vis = _colorize_labels(group_ids)
        init_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_3.png")
        pre_merge_path = init_seg_path
        pre_split_path = os.path.join(user_dir, f"{img_name}_pre_split.png")
        Image.fromarray(vis).save(init_seg_path)
        Image.fromarray(vis).save(pre_split_path)

        state = {
            "image": np.array(Image.open(rgba_path).convert("RGB")).tolist(),
            "processed_image": rgba_path,
            "group_ids": group_ids.tolist(),
            "original_group_ids": group_ids.tolist(),
            "img_name": img_name,
            "pre_split_path": pre_split_path,
            "save_mask_path": save_mask_path,
            "merged_seg_path": init_seg_path,
        }

        return init_seg_path, pre_merge_path, state

    size_th = threshold

    img = Image.open(image_path).convert("RGB")
    processed_image = prepare_image(img, rmbg_net=rmbg_model.to(DEVICE))

    processed_image = resize_and_pad_to_square(processed_image)
    white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
    white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
    image = np.array(white_bg_img.convert("RGB"))

    rgba_path = os.path.join(user_dir, f"{img_name}_processed.png")
    processed_image.save(rgba_path)

    print("Generating raw SAM masks without post-processing...")
    raw_masks = sam_mask_generator.generate(image)

    raw_sam_vis = np.copy(image)
    raw_sam_vis = np.ones_like(image) * 255

    sorted_masks = sorted(raw_masks, key=lambda x: x["area"], reverse=True)

    for i, mask_data in enumerate(sorted_masks):
        if mask_data["area"] < size_th:
            continue

        color_r = (i * 50 + 80) % 256
        color_g = (i * 120 + 40) % 256
        color_b = (i * 180 + 20) % 256
        color = np.array([color_r, color_g, color_b])

        mask = mask_data["segmentation"]
        raw_sam_vis[mask] = color

    visual = Visualizer(image)

    group_ids, pre_merge_im = get_sam_mask(
        image,
        sam_mask_generator,
        visual,
        merge_groups=None,
        rgba_image=processed_image,
        img_name=img_name,
        save_dir=user_dir,
        size_threshold=size_th,
    )

    pre_merge_path = os.path.join(user_dir, f"{img_name}_mask_pre_merge.png")
    Image.fromarray(pre_merge_im).save(pre_merge_path)
    pre_split_vis = np.ones_like(image) * 255

    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]

    for i, unique_id in enumerate(unique_ids):
        color_r = (i * 50 + 80) % 256
        color_g = (i * 120 + 40) % 256
        color_b = (i * 180 + 20) % 256
        color = np.array([color_r, color_g, color_b])

        mask = group_ids == unique_id
        pre_split_vis[mask] = color

        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            cv2.putText(
                pre_split_vis,
                str(unique_id),
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    pre_split_path = os.path.join(user_dir, f"{img_name}_pre_split.png")
    Image.fromarray(pre_split_vis).save(pre_split_path)
    print(
        f"Pre-split segmentation (before disconnected parts handling) saved to {pre_split_path}"
    )

    get_mask(group_ids, image, ids=2, img_name=img_name, save_dir=user_dir)

    init_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_2.png")

    seg_img = Image.open(init_seg_path)
    if seg_img.mode == "RGBA":
        white_bg = Image.new("RGBA", seg_img.size, (255, 255, 255, 255))
        seg_img = Image.alpha_composite(white_bg, seg_img)
        seg_img.save(init_seg_path)

    state = {
        "image": image.tolist(),
        "processed_image": rgba_path,
        "group_ids": (
            group_ids.tolist() if isinstance(group_ids, np.ndarray) else group_ids
        ),
        "original_group_ids": (
            group_ids.tolist() if isinstance(group_ids, np.ndarray) else group_ids
        ),
        "img_name": img_name,
        "pre_split_path": pre_split_path,
    }

    return init_seg_path, pre_merge_path, state


def apply_merge(merge_input, state, req: gr.Request):
    """Apply merge parameters and generate merged segmentation"""
    global sam_mask_generator

    if not state:
        return None, None, state

    # If we already have a merged mask (precomputed assets), skip SAM
    if state.get("merged_seg_path"):
        return state["merged_seg_path"], state

    user_dir = os.path.join(TMP_ROOT, str(req.session_hash), TAG_DIR)
    print("*" * 80, "User dir:", user_dir)
    # Convert back from list to numpy array
    image = np.array(state["image"])
    # Use original group IDs instead of the most recent ones
    group_ids = np.array(state["original_group_ids"])
    img_name = state["img_name"]

    # Load processed image from path
    processed_image = Image.open(state["processed_image"])

    # Display the original IDs before merging, SORTED for easier reading
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]  # Exclude background
    print(f"Original segment IDs (used for merging): {sorted(unique_ids.tolist())}")

    # Parse merge groups
    merge_groups = None
    try:
        if merge_input:
            merge_groups = []
            group_sets = merge_input.split(";")
            for group_set in group_sets:
                ids = [int(x) for x in group_set.split(",")]
                if ids:
                    # Validate if these IDs exist in the segmentation
                    existing_ids = [id for id in ids if id in unique_ids]
                    missing_ids = [id for id in ids if id not in unique_ids]

                    if missing_ids:
                        print(
                            f"Warning: These IDs don't exist in the segmentation: {missing_ids}"
                        )

                    # Only add group if it has valid IDs
                    if existing_ids:
                        merge_groups.append(ids)
                        print(
                            f"Valid merge group: {ids} (missing: {missing_ids if missing_ids else 'none'})"
                        )
                    else:
                        print(f"Skipping merge group with no valid IDs: {ids}")

            print(f"Using merge groups: {merge_groups}")
    except Exception as e:
        print(f"Error parsing merge groups: {e}")
        return None, None, state

    # Initialize visualizer
    visual = Visualizer(image)

    # Generate merged segmentation starting from original IDs
    # Add skip_split=True to prevent splitting after merging
    new_group_ids, merged_im = get_sam_mask(
        image,
        sam_mask_generator,
        visual,
        merge_groups=merge_groups,
        existing_group_ids=group_ids,
        rgba_image=processed_image,
        skip_split=True,
        img_name=img_name,
        save_dir=user_dir,
        size_threshold=size_th,
    )
    tgt_img_resolution_ = (518, 518)
    # Display the new IDs after merging for future reference
    new_unique_ids = np.unique(new_group_ids)
    new_unique_ids = new_unique_ids[new_unique_ids >= 0]  # Exclude background
    print(f"New segment IDs (after merging): {new_unique_ids.tolist()}")

    # Clean edges
    new_group_ids = clean_segment_edges(new_group_ids)

    # Save merged segmentation visualization
    get_mask(new_group_ids, image, ids=3, img_name=img_name, save_dir=user_dir)

    # Path to merged segmentation
    merged_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_3.png")

    save_mask = new_group_ids + 1
    save_mask = save_mask.reshape(518, 518, 1).repeat(3, axis=-1)
    cv2.imwrite(
        os.path.join(user_dir, f"{img_name}_mask.exr"), save_mask.astype(np.float32)
    )

    # Update state with the new group IDs but keep original IDs unchanged
    state["group_ids"] = (
        new_group_ids.tolist()
        if isinstance(new_group_ids, np.ndarray)
        else new_group_ids
    )
    state["save_mask_path"] = os.path.join(user_dir, f"{img_name}_mask.exr")

    return merged_seg_path, state


def explode_mesh(mesh, explosion_scale=0.4):

    if isinstance(mesh, trimesh.Scene):
        scene = mesh
    elif isinstance(mesh, trimesh.Trimesh):
        print("Warning: Single mesh provided, can't create exploded view")
        scene = trimesh.Scene(mesh)
        return scene
    else:
        print(f"Warning: Unexpected mesh type: {type(mesh)}")
        scene = mesh

    if len(scene.geometry) <= 1:
        print("Only one geometry found - nothing to explode")
        return scene

    print(f"[EXPLODE_MESH] Starting mesh explosion with scale {explosion_scale}")
    print(f"[EXPLODE_MESH] Processing {len(scene.geometry)} parts")

    exploded_scene = trimesh.Scene()

    part_centers = []
    geometry_names = []

    for geometry_name, geometry in scene.geometry.items():
        if hasattr(geometry, "vertices"):
            transform = scene.graph[geometry_name][0]
            vertices_global = trimesh.transformations.transform_points(
                geometry.vertices, transform
            )
            center = np.mean(vertices_global, axis=0)
            part_centers.append(center)
            geometry_names.append(geometry_name)
            print(f"[EXPLODE_MESH] Part {geometry_name}: center = {center}")

    if not part_centers:
        print("No valid geometries with vertices found")
        return scene

    part_centers = np.array(part_centers)
    global_center = np.mean(part_centers, axis=0)

    print(f"[EXPLODE_MESH] Global center: {global_center}")

    for i, (geometry_name, geometry) in enumerate(scene.geometry.items()):
        if hasattr(geometry, "vertices"):
            if i < len(part_centers):
                part_center = part_centers[i]
                direction = part_center - global_center

                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                else:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)

                offset = direction * explosion_scale
            else:
                offset = np.zeros(3)

            original_transform = scene.graph[geometry_name][0].copy()

            new_transform = original_transform.copy()
            new_transform[:3, 3] = new_transform[:3, 3] + offset

            exploded_scene.add_geometry(
                geometry, transform=new_transform, geom_name=geometry_name
            )

            print(
                f"[EXPLODE_MESH] Part {geometry_name}: moved by {np.linalg.norm(offset):.4f}"
            )

    print("[EXPLODE_MESH] Mesh explosion complete")
    return exploded_scene


def _prepare_urdf_package(export_root: str, outdir: str) -> str | None:
    """Create a URDF and zip bundle from the exported articulation directory."""

    json_candidates = glob(
        os.path.join(export_root, "**", "object.json"), recursive=True
    )
    if not json_candidates:
        print("No object.json found for URDF export.")
        return None

    target_json = Path(json_candidates[0])
    export_dir = target_json.parent
    urdf_path = export_dir / "object.urdf"

    try:
        config = ConverterConfig(
            json_path=target_json,
            output_path=urdf_path,
            asset_root=export_dir,
            mesh_priority="glb",
            geometry_format=None,
            absolute_mesh_paths=False,
            robot_name=None,
            min_mass=1.0,
            limit_effort=50.0,
            limit_velocity=1.0,
        )

        with config.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        robot_element, _ = build_robot_element(data, config)
        urdf_xml = prettify(robot_element)
        config.output_path.write_text(urdf_xml, encoding="utf-8")
    except Exception as exc:
        print(f"Failed to build URDF package: {exc}")
        return None

    zip_path = Path(outdir) / "pact_urdf_package.zip"
    try:
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for root, _, files in os.walk(export_dir):
                for filename in files:
                    abs_path = Path(root) / filename
                    rel_path = abs_path.relative_to(export_dir)
                    archive.write(abs_path, arcname=rel_path.as_posix())
    except Exception as exc:
        print(f"Failed to zip URDF package: {exc}")
        return None

    return zip_path.as_posix()


@spaces.GPU(duration=90)
def generate_parts(state, seed, cfg_strength, req: gr.Request):
    if pact_pipeline is None:
        raise RuntimeError("PAct pipeline is not loaded. Call prepare_models first.")

    if not state:
        return None, None, None

    explode_factor = 0.3
    session_root = os.path.join(TMP_ROOT, str(req.session_hash))
    user_dir = os.path.join(session_root, TAG_DIR)
    os.makedirs(user_dir, exist_ok=True)

    img_dataset = datasets.ImageConditioned_dataset(
        user_dir, is_depth_one_dir=True, image_size=518
    )
    if len(img_dataset) == 0:
        print("No processed samples found for this session.")
        return None, None, None
    dataloader = DataLoader(
        copy.deepcopy(img_dataset),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=img_dataset.collate_fn,
    )
    assert (
        len(img_dataset) == 1
    ), "Expected exactly one sample in the dataset for this session."
    outdir = os.path.join(user_dir, "pact_outputs")
    os.makedirs(outdir, exist_ok=True)

    arti_out_mode = "mean_feature_regression_steps"
    ss_steps = 25
    slat_steps = 25
    render_config = RenderConfig(
        num_frames=60,
        radius=2.3,
        fov=60.0,
        bg_color=(1.0, 1.0, 1.0),
    )
    export_config = ExportConfig(
        enabled=True,
        save_glb=True,
        save_gs=False,
        mesh_simplify_ratio=0.95,
        texture_size=1024,
        textured_mesh=True,
    )
    ss_sampler_params = {"steps": ss_steps, "cfg_strength": float(cfg_strength)}
    slat_sampler_params = {"steps": slat_steps, "cfg_strength": float(cfg_strength)}

    articulation_videos = []
    exploded_videos = []
    cond_images = []

    for batch_idx, batch in enumerate(dataloader):
        batch_result = pact_pipeline.run_inference_batch(
            batch,
            batch_index=batch_idx,
            seed=int(seed),
            device=torch.device(DEVICE),
            arti_out_mode=arti_out_mode,
            ss_params=ss_sampler_params,
            slat_params=slat_sampler_params,
            parts_explosion_scale=explode_factor,
            render_cfg=render_config,
            video_fps=20,
            save_individual_videos=False,
            save_conditional_images=False,
            export_cfg=export_config,
            dataset=img_dataset,
            outdir=outdir,
        )
        articulation_videos.extend(batch_result.articulation_videos)
        exploded_videos.extend(batch_result.exploded_videos)
        cond_images.extend(batch_result.cond_vis_images)

    arti_video_path = os.path.join(outdir, "arti_animation.mp4")
    exploded_video_path = os.path.join(outdir, "exploded_animation.mp4")
    imageio.mimsave(arti_video_path, batch_result.articulation_videos[0], fps=15)
    imageio.mimsave(exploded_video_path, batch_result.exploded_videos[0], fps=15)

    pact_pipeline.save_inference_grids(
        articulation_videos=articulation_videos,
        exploded_videos=exploded_videos,
        cond_images=cond_images,
        outdir=outdir,
        grid_size=4,
        video_fps=20,
        save_video_grid=False,
        save_cond_vis_grid=False,
    )

    export_root = os.path.join(outdir, "exported_arti_objects")
    urdf_zip_path = _prepare_urdf_package(export_root, outdir)

    return (arti_video_path, exploded_video_path, urdf_zip_path)
