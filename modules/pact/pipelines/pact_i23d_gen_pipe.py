from typing import *
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import os
import random

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, utils
from PIL import Image

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..modules.sparse.basic import SparseTensor, sparse_cat
from modules.utils.articulation_utils import convert_data_2_info, post_process_arti_info
from modules.utils.articulation_utils import (
    cat_ref,
    sem_ref,
    joint_ref,
    convert_json,
    convert_data_range,
    convert_data_2_info,
    post_process_arti_info,
    animate_and_render,
)
from modules.pact.datasets.structured_latent import export_arti_obj_to_singapo_style
from modules.pact.process_utils import (
    merge_multi_parts,
    make_slat_coords_from_voxel_coords,
    exploded_gaussians,
)
from modules.pact.utils import render_utils, postprocessing_utils


@dataclass
class RenderConfig:
    num_frames: int
    radius: float
    fov: float
    bg_color: Tuple[float, float, float]


@dataclass
class ExportConfig:
    enabled: bool
    save_glb: bool
    save_gs: bool
    mesh_simplify_ratio: float
    texture_size: int
    textured_mesh: bool


@dataclass
class BatchRunResult:
    articulation_videos: List[List[np.ndarray]]
    exploded_videos: List[np.ndarray]
    cond_vis_images: List[torch.Tensor]


class PActPipeline(Pipeline):
    """
    Pipeline for inferring OmniPart image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
        arti_out_mode: str = "regression_mean_steps",
        arti_mean_num: int = 20,
        postprocess_articulation: bool = False,
        verbose: bool = True,
        original_sampler: samplers.Sampler = None,
    ):
        # arti_out_mode = arti_out_mode
        self.arti_mean_num = arti_mean_num
        self.postprocess_articulation = postprocess_articulation
        self.verbose = verbose
        print(
            "config:",
            "\n" f"arti_out_mode={arti_out_mode}, ",
            "\n" f"arti_mean_num={arti_mean_num}, ",
            "\n" f"postprocess_articulation={postprocess_articulation}, ",
            "\n",
        )
        # Skip initialization if models is None (used in from_pretrained)
        # self.Arti_SLatVisMixin = type('Arti_SLatVisMixin', (object,), {
        if models is None:
            return

        super().__init__(models)
        self.is_decode_coords = True
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.original_sampler = original_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def __from_pretrained(path: str, revision: str = None) -> "Pipeline":

        import os
        import json
        from .. import models

        # Standard loading from directory or Hugging Face
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            print(
                f"Loading pipeline configuration from local path: {path}/pipeline.json"
            )
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download

            print(f"Downloading pipeline configuration from Hugging Face: {path}")
            config_file = hf_hub_download(path, "pipeline.json", revision=revision)
        # config_file  = "configs/pipeline/SSPart_SlatArt_combined_pipeline.json"
        ### print in red
        print(f"\033[31mLoading pipeline configuration from: {config_file}\033[0m")

        with open(config_file, "r") as f:
            args = json.load(f)["args"]

        print(f"loading models from {path}")
        _models = {}
        for k, v in args["models"].items():
            print(f"Loading model {k} from local path: {path}/{v}")

            if k != "slat_arti_flow_model":

                _models[k] = models.from_pretrained(f"{path}/{v}", revision=revision)
            else:
                _models[k] = models.from_pretrained_ArtiSLat(
                    f"{path}/{v}", revision=revision
                )
        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @staticmethod
    def from_pretrained(path: str, revision: str = None) -> "PActPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.

        Returns:
            PActPipeline: Loaded pipeline instance
        """
        pipeline = PActPipeline.__from_pretrained(path, revision=revision)
        new_pipeline = PActPipeline()
        new_pipeline.__dict__.update(pipeline.__dict__)
        args = pipeline._pretrained_args

        # Initialize samplers from saved arguments
        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.original_sampler = getattr(
            samplers, args["original_sampler"]["name"]
        )(**args["original_sampler"]["args"])
        new_pipeline.original_sampler_params = args["original_sampler"]["params"]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]
        new_pipeline._init_image_cond_model(args["image_cond_model"])
        # new_pipeline.arti_out_mode = args.get('arti_out_mode', new_pipeline.arti_out_mode)
        new_pipeline.arti_mean_num = args.get(
            "arti_mean_num", new_pipeline.arti_mean_num
        )
        new_pipeline.postprocess_articulation = args.get(
            "postprocess_articulation", new_pipeline.postprocess_articulation
        )
        # print(f"Loaded pipeline with arti_out_mode={new_pipeline.arti_out_mode}, arti_mean_num={new_pipeline.arti_mean_num}, postprocess_articulation={new_pipeline.postprocess_articulation}")
        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.

        Args:
            name (str): Name of the DINOv2 model to load
        """
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model

        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image_cond_model_transform = transform

    @staticmethod
    def _setup_rng(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _split_obj_parts(
        any_data_list: Sequence[Any], num_parts: Sequence[int]
    ) -> List[Any]:
        assert len(any_data_list) == sum(num_parts)
        start_idx = 0
        res = []
        for count in num_parts:
            res.append(any_data_list[start_idx])
            start_idx += count
        return res

    def preprocess_image(self, input: Image.Image, size=(518, 518)) -> Image.Image:
        """
        Preprocess the input image for the model.

        Args:
            input (Image.Image): Input image
            size (tuple): Target size for resizing

        Returns:
            Image.Image: Preprocessed image
        """
        img = np.array(input)
        if img.shape[-1] == 4:
            # Handle alpha channel by replacing transparent pixels with black
            mask_img = img[..., 3] == 0
            img[mask_img] = [0, 0, 0, 255]
            img = img[..., :3]
            img_rgb = Image.fromarray(img.astype("uint8"))
        # Resize to target size
        img_rgb = img_rgb.resize(size, resample=Image.Resampling.BILINEAR)
        return img_rgb

    @torch.no_grad()
    def encode_image(
        self, image: Union[torch.Tensor, List[Image.Image]]
    ) -> torch.Tensor:
        """
        Encode the image using the conditioning model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image(s) to encode

        Returns:
            torch.Tensor: The encoded features
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(
                isinstance(i, Image.Image) for i in image
            ), "Image list should be list of PIL images"
            # Convert PIL images to tensors
            if self.encoding_resolution is not None:
                image = [
                    i.resize(self.encoding_resolution, Image.LANCZOS) for i in image
                ]
            else:
                image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        # Apply normalization and run through DINOv2 model
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models["image_cond_model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, List[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: Dictionary with conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)  # Negative conditioning (zero)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_parts: torch.Tensor = None,
        sampler_params: dict = {},
        save_coords: bool = False,
        return_raw: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            save_coords (bool): Whether to save coordinates internally.

        Returns:
            torch.Tensor: Coordinates of the sparse structure
        """
        # Sample occupancy latent
        flow_model = self.models["sparse_structure_flow_model"]
        reso = (
            flow_model.resolution
            if hasattr(flow_model, "resolution")
            else flow_model.base_model.resolution
        )
        noise_channel = (
            flow_model.in_channels
            if hasattr(flow_model, "in_channels")
            else flow_model.base_model.in_channels
        )
        num_samples = (
            torch.sum(num_parts).item()
            if num_parts is not None
            else cond["cond"].shape[0]
        )
        noise = torch.randn(num_samples, noise_channel, reso, reso, reso).to(
            self.device
        )

        # Merge default and custom sampler parameters
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        # Generate samples using the sampler
        # if sampler_params["arti_out_mode"]==
        arti_out_mode = sampler_params.get("arti_out_mode", "regression_mean_steps")
        if arti_out_mode in ["flow_matching", "diffusion_head_feature_cache"]:
            arti_channel = 24
            noise_arti = torch.randn(num_samples, arti_channel).to(self.device)
            cond["noise_arti"] = noise_arti
        res = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
        )
        z_s = res.samples

        def average_dict_datasets(dict_list):
            outs = []
            arti_mean_num = min(self.arti_mean_num, len(dict_list))
            for dict_feats in dict_list[-arti_mean_num:]:
                assert (
                    dict_list[0].keys() == dict_feats.keys()
                ), "All dictionaries must have the same keys"
                outs.append(dict_feats["transformer_blocks"][-1])
            stacked = torch.stack(outs, dim=0)
            avg_feats = torch.mean(stacked, dim=0)
            avg_dict = {
                "transformer_blocks": [None]
                * (len(dict_list[0]["transformer_blocks"]) - 1)
                + [avg_feats]
            }
            return avg_dict

        arti = getattr(res, "arti", None)
        if arti is not None:

            if arti_out_mode == "mean_feature_regression_steps":
                pass
                feats = getattr(res, "pred_arti_t", None)

                # for key in dict_list[0].keys():

                # avg_dict = {}
                # for key in dict_list[0].keys():
                #     stacked = torch.stack([d[key] for d in dict_list], dim=0)
                #     avg_dict[key] = torch.mean(stacked, dim=0)
                # return avg_dict

                avg_info = average_dict_datasets(feats)
                res.pred_arti_t = None
                arti = flow_model.fwd_articulation_info(
                    avg_info,
                    **cond,
                    **sampler_params,
                )

            elif arti_out_mode == "flow_matching":
                pass  # Articulation is already in the correct format

            else:
                raise ValueError(f"Unknown arti_out_mode: {arti_out_mode}")

        # Decode occupancy latent to get coordinates
        if self.is_decode_coords:

            decoder = self.models["sparse_structure_decoder"]
            decoded_occ = decoder(z_s)
            coords = torch.argwhere(decoded_occ > 0)[:, [0, 2, 3, 4]].int()
        else:
            coords = None
            decoded_occ = None
        if save_coords:
            self.save_coordinates = coords

        if return_raw:
            return {
                "coords": coords,
                "latent": z_s,
                "occ": decoded_occ,
                "arti": arti,
                "mean_multi_steps_feature": (
                    avg_info["transformer_blocks"][-1]
                    if "avg_info" in locals()
                    else None
                ),
            }
        else:
            return coords, arti

    @torch.no_grad()
    def get_part_coords(
        self,
        image: Union[Image.Image, List[Image.Image]],
        masks: torch.Tensor = None,
        num_parts: torch.Tensor = None,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = True,
        save_coords: bool = False,
        return_raw: bool = True,
    ) -> dict:
        """
        Get coordinates of the sparse structure from an input image.

        Args:
            image: Input image or list of images
            num_samples: Number of samples to generate
            seed: Random seed
            sparse_structure_sampler_params: Additional parameters for the sparse structure sampler
            preprocess_image: Whether to preprocess the image
            save_coords: Whether to save coordinates internally

        Returns:
            torch.Tensor: Coordinates of the sparse structure
        """
        if isinstance(image, Image.Image):
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(
                cond, num_parts, sparse_structure_sampler_params, save_coords
            )
            return coords
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:
                cond = self.get_cond(image)
            else:
                cond = self.get_cond(image.unsqueeze(0))
            cond["ordered_mask_dino"] = masks
            cond["masks"] = masks
            cond["neg_mask"] = torch.zeros_like(cond["ordered_mask_dino"])
            cond["num_parts"] = num_parts

            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(
                cond,
                num_parts,
                sparse_structure_sampler_params,
                save_coords,
                return_raw,
            )

            # if "arti" in coords and coords["arti"] is not None:
            #     arti = coords["arti"]
            #     articulation = self.decode_ss_articulation(arti, num_parts)
            #     coords["articulation"] = articulation
            return coords
        elif isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(i) for i in image]
            cond = self.get_cond(image)
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(
                cond, num_parts, sparse_structure_sampler_params, save_coords
            )
            return coords
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

    def sample_slat_arti(
        self,
        cond: dict,
        coords: torch.Tensor,
        part_layouts: Optional[List[List[slice]]] = None,
        masks: torch.Tensor = None,
        sampler_params: dict = {},
        **kwargs,
    ) -> Tuple[sp.SparseTensor, Optional[sp.SparseTensor]]:
        # Sample structured latent
        flow_model = self.models["slat_arti_flow_model"]

        # Create noise tensor with same coordinates as the sparse structure
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.base_model.in_channels).to(
                self.device
            ),
            coords=coords,
        )

        # Merge default and custom sampler parameters
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        # Add part information if provided
        if part_layouts is not None:
            kwargs["part_layouts"] = part_layouts
        if masks is not None:
            kwargs["masks"] = masks
            kwargs["neg_mask"] = torch.zeros_like(masks)
        # kwargs["return_articulation"]= True if hasattr(self, 'is_predict_arti_info') and self.is_predict_arti_info else False ### NOTE: code Ominipart
        # kwargs["return_articulation"]= True  ### NOTE: code Ominipart

        # Generate samples
        res = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            **kwargs,  ## kwargs["masks"] = [N,37,37] kwargs["part_layouts"] = [[slice(0, 10), slice(10, 20), slice(20, 30)],...]
        )
        slat = res.samples

        arti = getattr(res, "arti", None)
        if arti is not None:
            arti_out_mode = sampler_params.get("arti_out_mode", "regression_mean_steps")

            if arti_out_mode == "regression_last_step":
                pass
            elif arti_out_mode == "regression_mean_steps":
                pred_seq = getattr(res, "pred_arti_t", None) or []
                arti_mean_num = min(self.arti_mean_num, len(pred_seq))
                if arti_mean_num > 0:
                    feats = [pre.feats for pre in pred_seq[-arti_mean_num:]]
                    feats = torch.mean(torch.stack(feats, dim=0), dim=0)
                    art_mean = sp.SparseTensor(coords=arti.coords, feats=feats)
                    assert art_mean.feats.shape == arti.feats.shape
                    arti = art_mean

            elif arti_out_mode == "mean_feature_regression_steps":
                feats = getattr(res, "pred_arti_t", None)

                def average_dict_datasets(dict_list):
                    outs = []
                    arti_mean_num = min(self.arti_mean_num, len(dict_list))
                    # print in red
                    print(
                        "\033[31mMean feature regression averaging over",
                        arti_mean_num,
                        "steps\033[0m",
                    )
                    # print("Mean feature regression averaging over", arti_mean_num, "steps")
                    for dict_feats in dict_list[-arti_mean_num:]:
                        assert (
                            dict_list[0].keys() == dict_feats.keys()
                        ), "All dictionaries must have the same keys"
                        outs.append(dict_feats["transformer_blocks"][-1].feats)
                    stacked = torch.stack(outs, dim=0)
                    avg_feats = torch.mean(stacked, dim=0)
                    avg_sparse = sp.SparseTensor(
                        coords=dict_list[0]["transformer_blocks"][-1].coords,
                        feats=avg_feats,
                    )
                    avg_dict = {
                        "transformer_blocks": [None]
                        * (len(dict_list[0]["transformer_blocks"]) - 1)
                        + [avg_sparse]
                    }
                    return avg_dict

                    # for key in dict_list[0].keys():

                    # avg_dict = {}
                    # for key in dict_list[0].keys():
                    #     stacked = torch.stack([d[key] for d in dict_list], dim=0)
                    #     avg_dict[key] = torch.mean(stacked, dim=0)
                    # return avg_dict

                avg_info = average_dict_datasets(feats)
                avg_info["part_wise_batch_ids"] = feats[0][
                    "part_wise_batch_ids"
                ]  ### NOTE: make sure the part_wise_batch_ids is correct
                avg_info["cond_img_msk"] = feats[0][
                    "cond_img_msk"
                ]  ### NOTE: make sure the part_wise_batch_ids is correct
                avg_info["t_emb"] = feats[0][
                    "t_emb"
                ]  ### NOTE: make sure the part_wise_batch_ids is correct
                arti = flow_model.fwd_articulation_info(
                    avg_info, **cond, **sampler_params, **kwargs
                )

            elif arti_out_mode == "diffusion":
                raise NotImplementedError("Diffusion arti_out_mode not implemented yet")
            else:
                raise ValueError(f"Unknown arti_out_mode: {arti_out_mode}")
        # Normalize the features
        feat_dim = slat.feats.shape[1]
        base_std = torch.tensor(self.slat_normalization["std"]).to(slat.device)
        base_mean = torch.tensor(self.slat_normalization["mean"]).to(
            slat.device
        )  ## torch.Size([8])

        # Handle different dimensionality cases
        if feat_dim == len(base_std):
            # Dimensions match, apply directly
            std = base_std[None, :]
            mean = base_mean[None, :]
        elif feat_dim == 8 and len(base_std) == 9:
            # Use first 8 dimensions when latent is 8-dimensional but normalization is 9-dimensional
            std = base_std[:8][None, :]
            mean = base_mean[:8][None, :]
            print(
                f"Warning: Normalizing {feat_dim}-dimensional features with first 8 dimensions of 9-dimensional parameters"
            )
        else:
            # Handle general case of dimension mismatch
            std = torch.ones((1, feat_dim), device=slat.device)
            mean = torch.zeros((1, feat_dim), device=slat.device)

            copy_dim = min(feat_dim, len(base_std))
            std[0, :copy_dim] = base_std[:copy_dim]
            mean[0, :copy_dim] = base_mean[:copy_dim]
            print(
                f"Warning: Feature dimensions mismatch. Using {copy_dim} dimensions for normalization"
            )

        # Apply normalization
        slat = slat * std + mean

        return slat, arti

    @torch.no_grad()
    def get_slat_arti(
        self,
        image: Union[Image.Image, List[Image.Image], torch.Tensor],
        coords: torch.Tensor,
        part_layouts: List[List[slice]],
        masks: torch.Tensor,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
    ) -> dict:
        if isinstance(image, Image.Image):
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
        elif isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(i) for i in image]
            cond = self.get_cond(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:
                cond = self.get_cond(image)
            elif image.ndim == 3:
                cond = self.get_cond(image.unsqueeze(0))
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        torch.manual_seed(seed)
        slat, arti = self.sample_slat_arti(
            cond, coords, part_layouts, masks, slat_sampler_params
        )
        slat = self.divide_slat(slat, part_layouts)
        decoded = self.decode_slat(slat, formats)
        articulation = (
            self.decode_articulation(arti, part_layouts, decoded)
            if arti is not None
            else None
        )
        if articulation is not None:
            decoded["articulation"] = articulation
        decoded["part_layouts"] = part_layouts
        decoded["masks"] = masks
        return decoded

    def run_inference_batch(
        self,
        batch: dict,
        *,
        batch_index: int,
        seed: int,
        device: torch.device,
        arti_out_mode: str,
        ss_params: dict,
        slat_params: dict,
        parts_explosion_scale: float,
        render_cfg: RenderConfig,
        video_fps: int,
        save_individual_videos: bool,
        save_conditional_images: bool,
        export_cfg: ExportConfig,
        dataset,
        outdir: str,
    ) -> BatchRunResult:
        """Run the two-stage inference pipeline for a single dataloader batch."""

        articulation_videos: List[List[np.ndarray]] = []
        exploded_videos: List[np.ndarray] = []
        cond_vis_images: List[torch.Tensor] = []

        num_parts_tensor = batch["num_parts"]
        num_parts = num_parts_tensor.cpu().numpy().tolist()

        with torch.inference_mode():
            self._setup_rng(seed)
            self.is_decode_coords = True
            sparse_params = {**ss_params}
            sparse_params.update(
                {
                    "arti_out_mode": arti_out_mode,
                    "part_idx": batch["part_idx"].to(device),
                }
            )
            raw_output = self.get_part_coords(
                batch["cond"].to(device),
                batch["masks"].to(device),
                num_parts=num_parts_tensor.to(device),
                seed=seed,
                return_raw=True,
                sparse_structure_sampler_params=sparse_params,
            )
            occ = raw_output["occ"]

        x_0_coords: List[torch.Tensor] = []
        x_0 = occ.to(device)
        for i in range(len(x_0)):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            x_0_coords.append(coords)
        x_0_merged = merge_multi_parts(x_0_coords, num_parts)

        _, _, slat_data_list = make_slat_coords_from_voxel_coords(x_0_coords, num_parts)

        coords = sparse_cat(
            [
                SparseTensor(
                    coords=slat_data_list[obj_idx]["coords"],
                    feats=torch.zeros(
                        slat_data_list[obj_idx]["coords"].shape[0], 1, device=device
                    ),
                )
                for obj_idx in range(len(slat_data_list))
            ]
        ).coords

        layouts = [
            slat_data_list[obj_idx]["part_layouts"]
            for obj_idx in range(len(slat_data_list))
        ]

        conds = batch["cond"].to(device)
        masks = batch["masks"].to(device)
        img_cond_vis = batch["img_mask_vis"]
        if conds.shape[0] == sum(num_parts):
            idx = torch.cumsum(torch.tensor([0] + num_parts), dim=0)
            idx = idx[: len(num_parts)]
            conds = conds[idx]
            masks = masks[idx]
            img_cond_vis = img_cond_vis[idx]

        formats = ["gaussian"] if not export_cfg.enabled else ["mesh", "gaussian"]
        slat_param_cfg = {**slat_params, "arti_out_mode": arti_out_mode}

        with torch.inference_mode():
            pact_output = self.get_slat_arti(
                conds,
                coords,
                layouts,
                masks,
                seed=seed,
                slat_sampler_params=slat_param_cfg,
                formats=formats,
                preprocess_image=False,
            )

        x_0_arti_list_slat = pact_output["articulation"]["info"]
        all_rep_parts_gs_list = pact_output["gaussian"]
        num_parts_slat = torch.tensor(num_parts) + 1
        idxs = torch.cumsum(torch.tensor([0] + num_parts_slat.tolist()), dim=0)
        obj_gs_ll = [
            all_rep_parts_gs_list[idxs[i] : idxs[i + 1]]
            for i in range(len(num_parts_slat))
        ]

        obj_mesh_ll = None  ### list of list of meshes for each part of each object, only if "mesh" in pact_output and export_cfg.enabled
        if "mesh" in pact_output:
            all_rep_parts_mesh_list = pact_output["mesh"]
            idxs_mesh = torch.cumsum(torch.tensor([0] + num_parts_slat.tolist()), dim=0)
            obj_mesh_ll = [
                all_rep_parts_mesh_list[idxs_mesh[i] : idxs_mesh[i + 1]]
                for i in range(len(num_parts_slat))
            ]

        obj_textured_mesh_ll = None
        if export_cfg.enabled and export_cfg.save_glb and obj_mesh_ll is not None:
            textured_meshes = [
                postprocessing_utils.to_glb(
                    all_rep_parts_gs_list[i],
                    all_rep_parts_mesh_list[i],
                    simplify=export_cfg.mesh_simplify_ratio,
                    texture_size=export_cfg.texture_size,
                    textured=export_cfg.textured_mesh,
                    transform_zup_to_yup=False,
                )
                for i in range(len(all_rep_parts_gs_list))
            ]
            obj_textured_mesh_ll = [
                textured_meshes[idxs[i] : idxs[i + 1]]
                for i in range(len(num_parts_slat))
            ]

        unique_idxs = self._split_obj_parts(batch["id"], num_parts)

        for obj_idx in range(len(x_0_merged)):
            rep_parts = obj_gs_ll[obj_idx]
            exploded_gs = exploded_gaussians(
                rep_parts[1:], explosion_scale=parts_explosion_scale
            )
            exploded_video = render_utils.render_video(
                exploded_gs,
                num_frames=render_cfg.num_frames,
                bg_color=render_cfg.bg_color,
                r=render_cfg.radius,
                fov=render_cfg.fov,
                verbose=False,
            )["color"]

            texture_mesh_list = None
            gs_list = None
            rep_parts_mesh = None
            meta_info = None

            if export_cfg.enabled and obj_mesh_ll is not None:
                obj_path = unique_idxs[obj_idx]
                obj_id = obj_path.split("/")[-1][:-3]
                index = obj_path[:-3]
                try:
                    meta = dataset.metadata.loc[index]
                except Exception:
                    meta = {"scale": 1.0, "offset": [0.0, 0.0, 0.0]}

                img_id = obj_path.split("/")[-1][-2:]
                class_id = index.split("/")[-2]

                meta_info = {
                    "img_id": img_id,
                    "class_id": class_id,
                    "obj_id": obj_id,
                    "scale": meta["scale"],
                    "offset": meta["offset"],
                    "out_dir": outdir,
                    "tag": "exported_arti_objects",
                }

                rep_parts_mesh = obj_mesh_ll[obj_idx]
                if export_cfg.save_glb and obj_textured_mesh_ll is not None:
                    texture_mesh_list = obj_textured_mesh_ll[obj_idx][1:]
                if export_cfg.save_gs:
                    gs_list = obj_gs_ll[obj_idx][1:]

            arti_ani_video = self.animate_with_articulation(
                [rep_parts[1:]],
                x_0_arti_list_slat[obj_idx : obj_idx + 1],
            )
            articulation_videos.extend(arti_ani_video)

            if export_cfg.enabled and rep_parts_mesh is not None and meta_info:
                self.export_arti_objects(
                    [rep_parts_mesh[1:]],
                    x_0_arti_list_slat[obj_idx : obj_idx + 1],
                    [meta_info],
                    texture_mesh_list=[texture_mesh_list],
                    gs_list=[gs_list],
                )

            if save_individual_videos and arti_ani_video:
                video_path = os.path.join(
                    outdir, f"{batch_index + obj_idx:02d}__articulation_animation.mp4"
                )
                imageio.mimsave(video_path, arti_ani_video[0], fps=video_fps)
                print(
                    "Articulation manipulation arti_ani_video saved to:",
                    video_path,
                )
                exploded_path = os.path.join(
                    outdir, f"{batch_index + obj_idx:02d}__exploded_part.mp4"
                )
                imageio.imwrite(exploded_path, exploded_video)

            if save_conditional_images:
                cond_path = os.path.join(
                    outdir, f"{batch_index + obj_idx:02d}__exploded_part.png"
                )
                imageio.imwrite(
                    cond_path,
                    (img_cond_vis[obj_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(
                        "uint8"
                    ),
                )

            exploded_videos.append(exploded_video)
            cond_vis_images.append(img_cond_vis[obj_idx])

        return BatchRunResult(
            articulation_videos=articulation_videos,
            exploded_videos=exploded_videos,
            cond_vis_images=cond_vis_images,
        )

    @staticmethod
    def _save_video_grid(
        video_list: List[np.ndarray], output_dir: str, filename: str, fps: int = 20
    ) -> None:
        if not video_list:
            return
        os.makedirs(output_dir, exist_ok=True)
        num_videos = len(video_list)
        grid_size = int(np.ceil(np.sqrt(num_videos)))
        frame_counts = [len(video) for video in video_list]
        max_frames = max(frame_counts)
        first_frame = video_list[0][0]
        frame_shape = first_frame.shape
        height, width = frame_shape[:2]
        channels = 1 if len(frame_shape) == 2 else frame_shape[2]
        dtype = first_frame.dtype
        grid_frames = []
        blank_frame = np.zeros(frame_shape, dtype=dtype)
        for frame_idx in range(max_frames):
            if channels == 1:
                grid_frame = np.zeros(
                    (grid_size * height, grid_size * width), dtype=dtype
                )
            else:
                grid_frame = np.zeros(
                    (grid_size * height, grid_size * width, channels), dtype=dtype
                )
            for video_idx, video in enumerate(video_list):
                row = video_idx // grid_size
                col = video_idx % grid_size
                top = row * height
                left = col * width
                if frame_idx < video.shape[0]:
                    frame = video[frame_idx]
                else:
                    frame = blank_frame
                if channels == 1:
                    grid_frame[top : top + height, left : left + width] = frame
                else:
                    grid_frame[top : top + height, left : left + width, :] = frame
            grid_frames.append(grid_frame)
        grid_path = os.path.join(output_dir, filename)
        imageio.mimsave(grid_path, grid_frames, fps=fps)

    def save_inference_grids(
        self,
        *,
        articulation_videos: List[List[np.ndarray]],
        exploded_videos: List[np.ndarray],
        cond_images: List[torch.Tensor],
        outdir: str,
        grid_size: int,
        video_fps: int,
        save_video_grid: bool,
        save_cond_vis_grid: bool,
    ) -> None:
        if save_video_grid:
            mode = "arti_animation"
            folder = os.path.join(outdir, f"grid_vids_samples_videos_{mode}")
            if not os.path.exists(folder):
                os.makedirs(folder)
            for idx in range(0, len(articulation_videos), grid_size * grid_size):
                sub_videos = articulation_videos[idx : idx + grid_size * grid_size]
                if not sub_videos:
                    continue
                self._save_video_grid(
                    [np.stack(video) for video in sub_videos],
                    folder,
                    f"{mode}_grid_sub{idx//(grid_size*grid_size)}.mp4",
                    fps=video_fps,
                )

            mode = "exploded_parts"
            folder = os.path.join(outdir, f"grid_vids_samples_videos_{mode}")
            if not os.path.exists(folder):
                os.makedirs(folder)
            for idx in range(0, len(exploded_videos), grid_size * grid_size):
                sub_videos = exploded_videos[idx : idx + grid_size * grid_size]
                if not sub_videos:
                    continue
                self._save_video_grid(
                    [np.stack(video) for video in sub_videos],
                    folder,
                    f"{mode}_grid_sub{idx//(grid_size*grid_size)}.mp4",
                    fps=video_fps,
                )

        if save_cond_vis_grid and cond_images:
            mode = "arti_animation"
            folder = os.path.join(outdir, f"grids_cond_vis_{mode}")
            if not os.path.exists(folder):
                os.makedirs(folder)
            num_images = len(cond_images)
            utils.save_image(
                cond_images,
                os.path.join(folder, f"grid_cond_vis_{mode}.png"),
                nrow=int(np.sqrt(num_images)),
                normalize=True,
                value_range=(0, 1),
            )

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent
            formats (List[str]): The formats to decode to

        Returns:
            dict: Decoded outputs in requested formats
        """
        ret = {}
        if "mesh" in formats:
            try:
                ret["mesh"] = self.models["slat_decoder_mesh"](slat)
            except Exception as e:
                print(f"Warning: Mesh decoding failed with error: {e}")
                ret["mesh"] = []
                with torch.no_grad():
                    for idx in range(slat.shape[0]):
                        ret["mesh"].extend(self.models["slat_decoder_mesh"](slat[idx]))
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)
            # ret['gaussian'] =
        if "radiance_field" in formats:
            ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def divide_slat(
        self,
        slat: SparseTensor,
        part_layouts: List[slice],
    ) -> List[SparseTensor]:
        """
        Divide the structured latent into parts.

        Args:
            slat (SparseTensor): The structured latent
            part_layouts (List[slice]): Layout information for parts

        Returns:
            SparseTensor: Processed and divided latent
        """
        sparse_part = []
        for part_id, part_layout in enumerate(part_layouts):
            for part_obj_id, part_slice in enumerate(part_layout):
                part_x_sparse_tensor = SparseTensor(
                    coords=slat[part_id].coords[
                        part_slice
                    ],  ### TODO: Batched version of slicing sparse tensor
                    feats=slat[part_id].feats[part_slice],
                )
                sparse_part.append(part_x_sparse_tensor)

        slat = sparse_cat(sparse_part)

        return self.remove_noise(slat)

    def remove_noise(self, z_batch):
        """
        clean last dimension if it is confidence.
        """
        # Create a new list for processed tensors
        processed_batch = []

        for i, z in enumerate(z_batch):
            coords = z.coords
            feats = z.feats

            # Only filter if features have a confidence dimension (9th dimension)
            if feats.shape[1] == 9:
                # Get the confidence values (last dimension)
                # last_dim = feats[:, -1]
                # sigmoid_val = torch.sigmoid(last_dim)

                # # Calculate filtering statistics
                # total_points = coords.shape[0]
                # to_keep = sigmoid_val >= 0.5
                # if self.remove_stage2_noise:
                #     kept_points = to_keep.sum().item()
                #     discarded_points = total_points - kept_points
                #     discard_percentage = (
                #         (discarded_points / total_points) * 100
                #         if total_points > 0
                #         else 0
                #     )

                #     if kept_points == 0:
                #         if self.verbose:
                #             print(f"No points kept for part {i}")
                #         continue

                #     if self.verbose:
                #         print(
                #             f"Discarded {discarded_points}/{total_points} points ({discard_percentage:.2f}%)"
                #         )
                # else:
                #     to_keep = torch.ones_like(sigmoid_val, dtype=torch.bool)
                # # Filter coordinates and features
                # coords = coords[to_keep]
                # feats = feats[to_keep]
                feats = feats[:, :-1]  # Remove the confidence dimension

                # Create a filtered SparseTensor
                processed_z = z.replace(coords=coords, feats=feats)
            else:
                processed_z = z

            processed_batch.append(processed_z)

        return sparse_cat(processed_batch)

    def decode_articulation(
        self,
        arti: sp.SparseTensor,
        part_layouts: List[List[slice]],
        decoded_formats: Optional[dict] = None,
    ) -> Optional[dict]:
        """Convert raw articulation predictions into structured metadata."""
        if arti is None or part_layouts is None:
            return None

        arti = arti.detach()
        batch_infos: List[List[dict]] = []
        batch_masks: List[Optional[np.ndarray]] = []

        for batch_idx in range(arti.shape[0]):
            batch_tensor = arti[batch_idx]
            feats = batch_tensor.feats
            if feats.numel() == 0:
                batch_infos.append([])
                batch_masks.append(None)
                continue

            preds_core = feats[:, :24].detach().cpu().numpy()
            info = convert_data_2_info(preds_core)
            batch_infos.append(info)

            if feats.shape[1] > 24:
                mask_prob = torch.sigmoid(feats[:, 24:]).detach().cpu().numpy()
                batch_masks.append(mask_prob)
            else:
                batch_masks.append(None)

        grouped_parts = self._group_decoded_parts(decoded_formats, part_layouts)
        post_processed: List[List[dict]] = []
        if grouped_parts is not None and self.postprocess_articulation:
            for infos, parts in zip(batch_infos, grouped_parts):
                try:
                    updated = post_process_arti_info(copy.deepcopy(infos), parts)
                except Exception as exc:
                    print(
                        f"Warning: post_process_arti_info failed ({exc}). Using raw articulation info."
                    )
                    updated = infos
                post_processed.append(updated)
        else:
            post_processed = batch_infos

        return {
            "sparse": arti,
            "info_raw": batch_infos,
            "info": post_processed,
            "mask": batch_masks,
            # 'arti_out_mode': arti_out_mode,
        }

    def _group_decoded_parts(
        self,
        decoded_formats: Optional[dict],
        part_layouts: List[List[slice]],
    ) -> Optional[List[List[Any]]]:
        if not decoded_formats:
            return None

        for key in ("gaussian", "mesh"):
            values = decoded_formats.get(key)
            if not isinstance(values, list):
                continue

            if len(values) == len(part_layouts) and all(
                isinstance(v, list) for v in values
            ):
                return values

            grouped: List[List[Any]] = []
            cursor = 0
            for layout in part_layouts:
                part_count = len(layout)
                sample_values = values[
                    cursor : cursor + part_count
                ]  ### add 1 to avoid full body part
                cursor += part_count
                grouped.append(sample_values)
            return grouped

        return None

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject
            num_images (int): The number of images to condition on
            num_steps (int): The number of steps to run the sampler for
            mode (str): Sampling strategy ('stochastic' or 'multidiffusion')
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f"_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            # Create schedule for which image to use at each step
            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(
                self,
                model,
                x_t,
                t,
                cond,
                neg_cond,
                cfg_strength,
                cfg_interval,
                **kwargs,
            ):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    # Average predictions from all conditions when within CFG interval
                    preds = []
                    for i in range(len(cond)):
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond[i : i + 1], **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(
                        self, model, x_t, t, neg_cond, **kwargs
                    )
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    # Average predictions from all conditions when outside CFG interval
                    preds = []
                    for i in range(len(cond)):
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond[i : i + 1], **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            # Restore original inference model
            sampler._inference_model = sampler._old_inference_model
            delattr(sampler, f"_old_inference_model")

    @torch.no_grad()
    def get_coords(
        self,
        image: Union[Image.Image, List[Image.Image]],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = True,
        save_coords: bool = False,
    ) -> dict:
        """
        Get coordinates of the sparse structure from an input image.

        Args:
            image: Input image or list of images
            num_samples: Number of samples to generate
            seed: Random seed
            sparse_structure_sampler_params: Additional parameters for the sparse structure sampler
            preprocess_image: Whether to preprocess the image
            save_coords: Whether to save coordinates internally

        Returns:
            torch.Tensor: Coordinates of the sparse structure
        """
        if isinstance(image, Image.Image):
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(
                cond, num_samples, sparse_structure_sampler_params, save_coords
            )
            return coords
        elif isinstance(image, torch.Tensor):
            cond = self.get_cond(image.unsqueeze(0))
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(
                cond, num_samples, sparse_structure_sampler_params, save_coords
            )
            return coords
        elif isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(i) for i in image]
            cond = self.get_cond(image)
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(
                cond, num_samples, sparse_structure_sampler_params, save_coords
            )
            return coords
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

    def animate_with_articulation(
        self, reps_parts_list: List[List], arti_info_list: List[dict]
    ):
        video_list = []
        is_postprocessing_arti_info = getattr(self, "is_postprocessing_arti_info", True)
        print("is_postprocessing_arti_info:", is_postprocessing_arti_info)
        for idx, (reps_parts, arti_info) in enumerate(
            zip(reps_parts_list, arti_info_list)
        ):

            if is_postprocessing_arti_info:
                arti_info = post_process_arti_info(arti_info, reps_parts)

            video = animate_and_render(reps_parts, arti_info)
            video_list.append(video)
        return video_list

    def export_arti_objects(
        self,
        reps_parts_list: List[List],
        arti_info_list: List[dict],
        path_info_list: List[dict],
        texture_mesh_list: Optional[List[List]] = None,
        gs_list: Optional[List[List]] = None,
    ):
        rot_matrix_inv = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )  ## to blender formats
        rot_matrix = np.linalg.inv(rot_matrix_inv)

        is_postprocessing_arti_info = getattr(self, "is_postprocessing_arti_info", True)
        print("is_postprocessing_arti_info:", is_postprocessing_arti_info)
        for idx, (reps_parts, arti_info, path_info) in enumerate(
            zip(reps_parts_list, arti_info_list, path_info_list)
        ):

            if is_postprocessing_arti_info:
                arti_info = post_process_arti_info(arti_info, reps_parts)
            out_dir = path_info["out_dir"]
            img_id = path_info["img_id"]
            obj_id = path_info["obj_id"]
            class_id = path_info["class_id"]
            tag = path_info["tag"]
            scale = path_info["scale"]
            shift = path_info["offset"]
            if scale is None or shift is None:
                scale_shift = None
            else:
                scale_shift = (scale, shift)

            export_dir = os.path.join(out_dir, tag, f"{class_id}@{obj_id}@{img_id}")
            try:
                export_arti_obj_to_singapo_style(
                    reps_parts,
                    arti_info,
                    export_dir=export_dir,
                    transfor_matrix=rot_matrix,
                    scale_shift=scale_shift,
                    textured_mesh_parts=(
                        texture_mesh_list[idx]
                        if texture_mesh_list is not None
                        else None
                    ),
                    gs_parts=gs_list[idx] if gs_list is not None else None,
                )
            except Exception as e:
                print("Error in exporting singapo style for :", export_dir, e)

        return
