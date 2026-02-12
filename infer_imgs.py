import os
import sys
import json
import imageio
import copy
import torch
import argparse
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from typing import *

import tqdm
import random
import numpy as np
from modules.pact import datasets

from argparse import BooleanOptionalAction
from modules.pact.pipelines import (
    PActPipeline,
    RenderConfig,
    ExportConfig,
)


os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def normalize_rgb(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("render_bg_color expects exactly three values (R, G, B).")
    return tuple(float(v) for v in values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, help="Experiment config file"
    )
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size for inference"
    )

    parser.add_argument(
        "--export_arti_objects", action="store_true", help="export to singapore format"
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        help="save_textured glbs of parts, time consuming!",
    )
    parser.add_argument(
        "--save_gs",
        action="store_true",
        help="save_textured glbs of parts, time consuming!",
    )
    parser.add_argument(
        "--arti_mean_num",
        type=int,
        default=20,
        help="Number of mean articulation samples to use",
    )
    parser.add_argument(
        "--arti_out_mode",
        type=str,
        default="mean_feature_regression_steps",  ### now only support mean_feature_regression_steps, can add more modes like flow_matching, diffusion_head_feature_cache, regression_mean_steps
        choices=[
            "mean_feature_regression_steps",
            "flow_matching",
            "regression_mean_steps",
        ],
        help="Batch size for inference",
    )

    parser.add_argument(
        "--slat_cfg_strength",
        type=float,
        default=7.0,
        help="Classifier-free guidance strength used during SLAT sampling",
    )
    parser.add_argument(
        "--ss_cfg_strength",
        type=float,
        default=7.0,
        help="Classifier-free guidance strength used during sparse-structure sampling",
    )
    parser.add_argument(
        "--ss_steps",
        type=int,
        default=25,
        help="Number of sampling steps for sparse-structure sampling",
    )
    parser.add_argument(
        "--slat_steps",
        type=int,
        default=25,
        help="Number of sampling steps for SLAT sampling",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=4,
        help="Grid dimension when saving video/image mosaics",
    )
    parser.add_argument(
        "--save_video_grid",
        action=BooleanOptionalAction,
        default=True,
        help="Enable or disable saving articulation/explosion video grids",
    )
    parser.add_argument(
        "--save_cond_vis_grid",
        action=BooleanOptionalAction,
        default=True,
        help="Enable or disable saving conditioning image grids",
    )
    parser.add_argument(
        "--explode_coords_ratio",
        type=float,
        default=0.5,
        help="Explosion ratio applied to voxel coordinates for visualization",
    )
    parser.add_argument(
        "--parts_explosion_scale",
        type=float,
        default=0.3,
        help="Explosion ratio applied to Gaussian primitives for visualization",
    )
    parser.add_argument(
        "--render_num_frames",
        type=int,
        default=60,
        help="Number of frames rendered for each Gaussian animation",
    )
    parser.add_argument(
        "--render_radius",
        type=float,
        default=2.3,
        help="Orbit camera radius for Gaussian rendering",
    )
    parser.add_argument(
        "--render_fov",
        type=float,
        default=60.0,
        help="Field of view (degrees) for Gaussian rendering",
    )
    parser.add_argument(
        "--render_bg_color",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help="Background color for Gaussian rendering (range 0-1)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=20,
        help="Frames per second for saved videos",
    )
    parser.add_argument(
        "--mesh_simplify_ratio",
        type=float,
        default=0.95,
        help="Mesh simplification ratio when exporting GLBs",
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=1024,
        help="Texture resolution when exporting GLBs",
    )
    parser.add_argument(
        "--textured_mesh",
        action=BooleanOptionalAction,
        default=True,
        help="Enable or disable textured mesh export",
    )

    cmd = sys.executable + " " + " \\ \n".join(sys.argv)

    args = parser.parse_args()

    # config = json.load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Combine arguments and config
    cfg = edict()
    cfg.update(args.__dict__)
    # cfg.update(config)

    pact_pipeline = PActPipeline.from_pretrained("PAct000/PAct", revision="main")

    articulation_ani_videos = []
    exploded_parts_videos = []
    cond_imgs_vis = []
    arti_out_mode = cfg.get("arti_out_mode", args.arti_out_mode)
    is_save_video_grid = cfg.get("save_video_grid", True)
    is_save_cond_vis_grid = cfg.get("save_cond_vis_grid", True)
    grid_size_max = cfg.get("grid_size", 4)
    slat_cfg_strength = cfg.get("slat_cfg_strength", 7.0)
    ss_cfg_strength = cfg.get("ss_cfg_strength", 7.0)
    ss_steps = cfg.get("ss_steps", 25)
    slat_steps = cfg.get("slat_steps", 25)
    video_fps = cfg.get("video_fps", 20)
    render_bg_color_cfg = cfg.get("render_bg_color", (1.0, 1.0, 1.0))
    if render_bg_color_cfg is None:
        render_bg_color_cfg = (1.0, 1.0, 1.0)
    render_bg_color = normalize_rgb(render_bg_color_cfg)
    render_num_frames = cfg.get("render_num_frames", 60)
    render_radius = cfg.get("render_radius", 2.3)
    render_fov = cfg.get("render_fov", 60.0)
    explode_coords_ratio = cfg.get("explode_coords_ratio", 0.5)
    parts_explosion_scale = cfg.get("parts_explosion_scale", 0.3)
    mesh_simplify_ratio = cfg.get("mesh_simplify_ratio", 0.95)
    texture_size = cfg.get("texture_size", 1024)
    textured_mesh = cfg.get("textured_mesh", True)
    seed = cfg.get("seed", 42)
    setup_rng(seed)

    args.outdir = os.path.join(
        args.outdir,
        f"seed{seed}_slatcfg{slat_cfg_strength}_sscfg{ss_cfg_strength}_sssteps{ss_steps}_slatsteps{slat_steps}_artiout{arti_out_mode}",
    )

    pact_pipeline.verbose = False

    print(args.outdir)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    with open(os.path.join(args.outdir, "run_command.txt"), "w") as f:
        f.write(cmd + "\n")

    pact_pipeline.to(device)
    pact_pipeline.arti_mean_num = cfg.get("arti_mean_num", args.arti_mean_num)

    batch_size = cfg.get("batch_size", 1)
    explode_coords_ratio = float(explode_coords_ratio)
    parts_explosion_scale = float(parts_explosion_scale)

    render_config = RenderConfig(
        num_frames=render_num_frames,
        radius=render_radius,
        fov=render_fov,
        bg_color=render_bg_color,
    )
    export_config = ExportConfig(
        enabled=args.export_arti_objects,
        save_glb=args.save_glb,
        save_gs=args.save_gs,
        mesh_simplify_ratio=mesh_simplify_ratio,
        texture_size=texture_size,
        textured_mesh=textured_mesh,
    )
    ss_sampler_params = {"steps": ss_steps, "cfg_strength": ss_cfg_strength}
    slat_sampler_params = {"steps": slat_steps, "cfg_strength": slat_cfg_strength}
    ### PM-Dataset
    # img_dataset = getattr(datasets, "ImageConditionedPartBasedSparseStructureLatent")(
    #     cfg.data_dir, is_test=cfg.test_only, **cfg.ss_dataset.args
    # )
    ### Real-Dataset

    img_dataset = getattr(datasets, "ImageConditioned_dataset")(
        "assets/real_world_examples",
    )
    dataloader = DataLoader(
        copy.deepcopy(img_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No parallelization for simplicity
        collate_fn=(
            img_dataset.collate_fn if hasattr(img_dataset, "collate_fn") else None
        ),
    )

    data_iterator = iter(dataloader)

    # for batch_idx in tqdm.tqdm(range(0, 6, batch_size)):
    for batch_idx in tqdm.tqdm(range(0, len(img_dataset), batch_size)):
        print("Processing batch ", batch_idx)
        data = next(data_iterator)
        batch_result = pact_pipeline.run_inference_batch(
            data,
            batch_index=batch_idx,
            seed=seed,
            device=device,
            arti_out_mode=arti_out_mode,
            ss_params=ss_sampler_params,
            slat_params=slat_sampler_params,
            parts_explosion_scale=parts_explosion_scale,
            render_cfg=render_config,
            video_fps=video_fps,
            save_individual_videos=not is_save_video_grid,
            save_conditional_images=not is_save_cond_vis_grid,
            export_cfg=export_config,
            dataset=img_dataset,
            outdir=args.outdir,
        )
        articulation_ani_videos.extend(batch_result.articulation_videos)
        exploded_parts_videos.extend(batch_result.exploded_videos)
        cond_imgs_vis.extend(batch_result.cond_vis_images)

    pact_pipeline.save_inference_grids(
        articulation_videos=articulation_ani_videos,
        exploded_videos=exploded_parts_videos,
        cond_images=cond_imgs_vis,
        outdir=args.outdir,
        grid_size=grid_size_max,
        video_fps=video_fps,
        save_video_grid=is_save_video_grid,
        save_cond_vis_grid=is_save_cond_vis_grid,
    )
