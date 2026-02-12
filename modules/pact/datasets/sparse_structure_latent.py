import os
import json
from typing import *
import concurrent
import numpy as np
import torch
from tqdm import tqdm
import utils3d

from modules.inference_utils import change_pcd_range
from modules.pact.process_utils import exploded_coords
from modules.pact.utils import render_utils
from modules.pact.utils.data_utils import load_balanced_group_indices
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import (
    StandardDatasetBase,
    TextConditionedMixin,
    ImageConditionedMixin,
    StandardDatasetBase_Part,
    ImageConditionedMixin_PartNet,
)
from .. import models
import copy


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        self.vis_video = self.kwargs.get("vis_video", True)

    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(
                **cfg["models"]["decoder"]["args"]
            )
            ckpt_path = os.path.join(
                self.ss_dec_path, "ckpts", f"decoder_{self.ss_dec_ckpt}.pt"
            )
            decoder.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True)
            )
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i : i + batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0["x_0"]
        x_0 = self.decode_latent(x_0.cuda())

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = "voxel"

        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        ### set random seed for consistent visualization
        # np.random.seed(42) ## TODO: make seed configurable
        # yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws_offset = 0
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        radius = 2.4
        for yaw, pitch in zip(yaws, pitch):
            orig = (
                torch.tensor(
                    [
                        np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch),
                        np.sin(pitch),
                    ]
                )
                .float()
                .cuda()
                * radius
            )
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []

        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device="cuda",
                primitive="voxel",
                sh_degree=0,
                primitive_config={"solid": True},
            )
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(resolution)),
                dtype=torch.uint8,
                device="cuda",
            )

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(
                    representation, ext, intr, colors_overwrite=representation.position
                )
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)

        return torch.stack(images)


class PartBased_SparseStructureLatentVisMixin(SparseStructureLatentVisMixin):

    @torch.no_grad()
    def visualize_sample(self, x_0_: Union[torch.Tensor, dict]):
        assert isinstance(
            x_0_, dict
        ), "For part-based sparse structure latent, x_0 should be a dict containing 'x_0' and 'num_parts'"
        x_0 = x_0 if isinstance(x_0_, torch.Tensor) else x_0_["x_0"]
        num_parts = x_0_["num_parts"]
        x_0 = self.decode_latent(x_0.cuda())

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = "voxel"

        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        ### set random seed for consistent visualization
        # np.random.seed(42) ## TODO: make seed configurable
        # yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws_offset = 0
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        radius = 2.3
        for yaw, pitch in zip(yaws, pitch):
            orig = (
                torch.tensor(
                    [
                        np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch),
                        np.sin(pitch),
                    ]
                )
                .float()
                .cuda()
                * radius
            )
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        def merge_multi_parts(x_0, num_parts, explode_ratio=0.0):
            s = 0
            merged_x0 = []
            for num_part in num_parts:
                x_0_part = x_0[s : s + num_part]

                if explode_ratio > 0.0:
                    x_0_part = exploded_coords(x_0_part, explosion_scale=explode_ratio)
                    coord_rg = (x_0_part.min(), x_0_part.max())
                    x_0_part = change_pcd_range(
                        x_0_part, from_rg=coord_rg, to_rg=(1e-3, 1 - 1e-3)
                    )
                else:
                    x_0_part = torch.cat(x_0_part, dim=0).cuda()
                s += num_part

                merged_x0.append(x_0_part)

            return merged_x0

        x_0_coords = []
        x_0 = x_0.cuda()
        resolution = x_0.shape[-1]
        for i in range(len(x_0)):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            coords = coords.float() / resolution
            x_0_coords.append(coords)
        x_0_merged = merge_multi_parts(x_0_coords, num_parts, explode_ratio=0.3)

        representations = []
        # Build each representation once so we can reuse it for both image grids and videos
        for coords in x_0_merged:
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device="cuda",
                primitive="voxel",
                sh_degree=0,
                primitive_config={"solid": True},
            )
            representation.position = coords
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(resolution)),
                dtype=torch.uint8,
                device="cuda",
            )
            representations.append(representation)

        images = []
        tile = [2, 2]
        for representation in representations:
            image = torch.zeros(3, 1024, 1024, device="cuda")
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(
                    representation, ext, intr, colors_overwrite=representation.position
                )
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)

        videos = None
        if self.vis_video:
            videos = []
            video_cfg = {
                "num_frames": self.kwargs.get("vis_video_frames", 60),
                "bg_color": self.kwargs.get("vis_video_bg_color", (1, 1, 1)),
                "r": self.kwargs.get("vis_video_radius", 2.4),
                "fov": self.kwargs.get("vis_video_fov", 60),
                "resolution": self.kwargs.get("vis_video_resolution", 512),
            }
            for representation in representations:
                video = render_utils.render_video(
                    representation,
                    resolution=video_cfg["resolution"],
                    bg_color=video_cfg["bg_color"],
                    num_frames=video_cfg["num_frames"],
                    r=video_cfg["r"],
                    fov=video_cfg["fov"],
                    verbose=False,
                    colors_overwrite=representation.position,
                )["color"]
                videos.append(video)

        images = torch.stack(images)
        if self.vis_video:
            return images, videos
        return images, None


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f"ss_latent_{self.latent_model}"]]
        stats["With sparse structure latents"] = len(metadata)
        metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
        stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        latent = np.load(
            os.path.join(root, "ss_latents", self.latent_model, f"{instance}.npz")
        )
        z = torch.tensor(latent["mean"]).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            "x_0": z,
        }
        return pack


from modules.utils.part_utils import prepaer_parts_meta


# class PartBased_SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase_Part):
class PartBased_SparseStructureLatent(
    PartBased_SparseStructureLatentVisMixin, StandardDatasetBase_Part
):
    """
    Sparse structure latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        verbose: bool = False,
        is_test: bool = False,
        split_info_json: Optional[str] = None,
        **kwargs,
    ):
        self.latent_model = latent_model
        self.normalization = normalization
        self.value_range = (0, 1)

        self.verbose = verbose

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)
        self.kwargs = kwargs
        self.render_folder = self.kwargs.get(
            "render_folder", "render_merged_fixed_eevee"
        )
        self.latent_folder = self.kwargs.get("latent_folder", "ss_latents")
        self.latent_model_name = self.kwargs.get(
            "latent_model_name", "ss_enc_conv3d_16l8_fp16"
        )
        self.voxel_folder = self.kwargs.get("voxel_folder", "voxels_merged_fixed")
        self.img_cond_folder = self.kwargs.get("img_cond_folder", "imgs")
        self.msk_cond_folder = self.kwargs.get(
            "msk_cond_folder", "imgs/semantic_masks_merge_fixed"
        )
        self.preprocess_folder = self.kwargs.get(
            "preprocess_folder", "trellis_part_preprocess"
        )
        self.is_permute_latent_order = self.kwargs.get("is_permute_latent_order", False)
        self.is_predict_arti_info = self.kwargs.get(
            "is_predict_arti_info", False
        )  ### TODO: NOTE:
        self.is_mask_primistic_axis_o = self.kwargs.get(
            "is_mask_primistic_axis_o", False
        )  ### TODO: NOTE:

        self.part_info_file = (
            "object_merge_fixed.json"
            if self.kwargs.get("merge_fixed_part", True)
            else "object.json"
        )
        self.transform_info_file = os.path.join(
            "trellis_part_preprocess", self.render_folder, "full/transforms.json"
        )

        super().__init__(
            roots,
            is_test=is_test,
            split_info_json=split_info_json,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        self.data_cache_ss_part = {}
        paths = [path for _, path in self.instances]

        n_workers = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda p: PartBased_SparseStructureLatent.get_instance(self, p),
                        paths,
                    ),
                    total=len(paths),
                    desc=f"Preloading instances (workers={n_workers})",
                )
            )

            ###
        self.instance_parts = [
            (
                self.data_cache_ss_part[instant_path]["num_parts"]
                if self.data_cache_ss_part[instant_path]
                else 0
            )
            for _, instant_path in self.instances
        ]
        ## list of np.int64
        print(
            "max loads",
            max(self.instance_parts),
            "\n",
            "min load",
            min(self.instance_parts),
        )

        # self.instance_parts = [self.data_cache_slat_part[instant_path]["coords"].shape[0] if self.data_cache_slat_part[instant_path] else 0 for _, instant_path in enumerate(self.instances)]

        if not is_test:
            ##
            filtered_obj = []
            for idx, (root, path) in enumerate(self.instances):
                if self.instance_parts[idx] > 8:  ## TODO: make max_load configurable
                    pass
                else:
                    filtered_obj.append((root, path))
            self.instances = filtered_obj

        print("after filtered", len(self.instances))
        self.instance_parts = [
            (
                self.data_cache_ss_part[instant_path]["num_parts"]
                if self.data_cache_ss_part[instant_path]
                else 0
            )
            for _, instant_path in self.instances
        ]

    def load_partbased_ss_latent(
        self, object_path, part_info_sorted=None, is_sorted_by_id: bool = False
    ):
        """_summary_

        Args:
            object_path (_type_): _description_
            part_info_sorted (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        base_dir = object_path
        if part_info_sorted is None:

            meta_info_path = os.path.join(base_dir, "object_merge_fixed.json")
            raise NotImplementedError
        # transform_info_path = os.path.join(base_dir,f"trellis_part_preprocess/{self.render_folder}/full","transforms.json")
        latent_list = []
        for part in part_info_sorted:
            if self.verbose:
                print(part["name"], part["id"])
            part_suffix = "part_" + str(part["id"]) + "_" + part["name"]

            part_latent_path = os.path.join(
                base_dir,
                f"trellis_part_preprocess/{self.latent_folder}/{self.latent_model_name}/{part_suffix}.npz",
            )
            part_latent = np.load(part_latent_path)
            z = torch.tensor(part_latent["mean"]).float()
            if self.normalization is not None:
                z = (z - self.mean) / self.std

            latent_list.append(
                {
                    "suffix": part_suffix,
                    "latents": z,
                    "id": part["id"],
                }
            )
        full_parts_latent = np.load(
            os.path.join(
                base_dir,
                f"trellis_part_preprocess/{self.latent_folder}/{self.latent_model_name}/full.npz",
            )
        )
        full_parts_latent = torch.tensor(full_parts_latent["mean"]).float()
        if self.normalization is not None:
            full_parts_latent = (full_parts_latent - self.mean) / self.std
        if is_sorted_by_id:
            latent_list = sorted(latent_list, key=lambda x: x["id"])
        ### insert full latent at the beginning
        latent_list.insert(
            0,
            {
                "suffix": "full",
                "latents": full_parts_latent,
                "id": -1,
            },
        )

        return latent_list

    def get_instance(self, instance_path):

        if instance_path in self.data_cache_ss_part:
            if self.is_permute_latent_order:
                pass
                data_ins = copy.deepcopy(self.data_cache_ss_part[instance_path])
                permute_idx = torch.randperm(data_ins["num_parts"])
                data_ins["x_0"] = data_ins["x_0"][permute_idx]
                if self.is_predict_arti_info:
                    data_ins["x_0_arti"] = data_ins["x_0_arti"][permute_idx]
                    data_ins["x_0_arti_mask"] = data_ins["x_0_arti_mask"][permute_idx]
                data_ins["part_idx"] = data_ins["part_idx"][permute_idx]

                return data_ins

            else:
                return self.data_cache_ss_part[instance_path]
        try:

            part_info_sorted, bbox = prepaer_parts_meta(
                instance_path,
                part_info_path=self.part_info_file,
                transform_info_path=self.transform_info_file,
            )
            ###
            latent_list = self.load_partbased_ss_latent(instance_path, part_info_sorted)
            if self.is_predict_arti_info:

                def info_2_arti_array(part_info_sorted):
                    arts = []
                    arts_mask = []
                    for part in part_info_sorted:
                        node_data, mask = self._prepare_node_data(part)
                        arts.append(node_data)
                        arts_mask.append(mask)
                    return np.array(arts, dtype=np.float32), np.array(
                        arts_mask, dtype=np.float32
                    )

                arti_info, arti_info_mask = info_2_arti_array(part_info_sorted)
                arti_info = torch.tensor(arti_info, dtype=torch.float32)
                arti_info_mask = torch.tensor(arti_info_mask, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading data for instance {instance_path}: {e}")
            # raise e
        # latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))

        pack = {
            # 'x_0': latent_list[0]['latents'], ##  ### C,H,W,D
            "x_0": torch.stack(
                [latent["latents"] for latent in latent_list[1:]]
            ),  ### num_parts,C,H,W,D
            "num_parts": len(latent_list) - 1,
            "x_0_arti": arti_info if self.is_predict_arti_info else None,
            "x_0_arti_mask": arti_info_mask if self.is_predict_arti_info else None,
            "part_idx": torch.tensor([latent["id"] for latent in latent_list[1:]]),
        }
        self.data_cache_ss_part[instance_path] = pack

        return pack

    @staticmethod
    def collate_fn_(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]  #  [[0, 1]， [2, 3]]
        else:
            pass
            # group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            packs.append(sub_batch)

        if split_size is None:
            return packs[0]
        return packs

        batch_size = len(batch)
        x_0_list = [item["x_0"].unsqueeze(0) for item in batch]
        x_0 = torch.cat(x_0_list, dim=0)
        pack = {
            "x_0": x_0,
        }
        return pack

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]  #  [[0, 1]， [2, 3]]
        else:
            group_idx = load_balanced_group_indices(
                [b["num_parts"] for b in batch], split_size
            )
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            latents = []
            artis = []
            num_parts = []
            pre_cached_feature = []
            part_indexes = []
            start = 0
            start_art = 0
            for i, b in enumerate(sub_batch):
                x_0_i = b["x_0"]
                num_parts_i = b["num_parts"]
                part_indexes.append(b["part_idx"])
                latents.append(x_0_i)
                num_parts.append(num_parts_i)
                if "x_0_arti" in b and b["x_0_arti"] is not None:
                    artis.append(torch.cat([b["x_0_arti"], b["x_0_arti_mask"]], dim=-1))
                    # artis.append(b["x_0_arti"])
                if "pre_cached_feature" in b and b["pre_cached_feature"] is not None:
                    pre_cached_feature.append(b["pre_cached_feature"])

            pack["x_0"] = torch.cat(latents)
            pack["num_parts"] = torch.tensor(num_parts).to(torch.int32)
            pack["part_idx"] = torch.cat(part_indexes)
            if len(pre_cached_feature) > 0:
                pack["pre_cached_feature"] = torch.cat(pre_cached_feature)
            if len(artis) > 0:
                pack["x_0_arti"] = torch.cat(artis)

            assert (
                torch.sum(pack["num_parts"]) == pack["x_0"].shape[0]
            ), f"num_parts sum should be equal to x_0 shape[0]! {torch.sum(pack['num_parts'])}, {pack['x_0'].shape[0]}"
            # collate sparse tensor

            keys = [
                k
                for k in sub_batch[0].keys()
                if k
                not in [
                    "part_idx",
                    "pre_cached_feature",
                    "x_0",
                    "num_parts",
                    "x_0_arti",
                    "x_0_arti_mask",
                ]
            ]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.cat(
                        [
                            torch.stack(
                                [
                                    b[k],
                                ]
                                * b["num_parts"]
                            )
                            for b in sub_batch
                        ]
                    )
                    assert (
                        pack[k].shape[0] == pack["x_0"].shape[0]
                    ), f"collated {k} shape[0] should be equal to x_0 shape[0]! {pack[k].shape[0]}, {pack['x_0'].shape[0]}"
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = [b[k] for b in sub_batch]
                    raise NotImplementedError("list type not supported yet!")
                else:
                    pack[k] = [
                        [
                            b[k],
                        ]
                        * b["num_parts"]
                        for b in sub_batch
                    ]
                    pack[k] = sum(pack[k], [])

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class TextConditionedSparseStructureLatent(TextConditionedMixin, SparseStructureLatent):
    """
    Text-conditioned sparse structure dataset
    """

    pass


class ImageConditionedSparseStructureLatent(
    ImageConditionedMixin, SparseStructureLatent
):
    """
    Image-conditioned sparse structure dataset
    """

    pass


class ImageConditionedPartBasedSparseStructureLatent(
    ImageConditionedMixin_PartNet, PartBased_SparseStructureLatent
):
    """
    Image-conditioned sparse structure dataset
    """

    pass
