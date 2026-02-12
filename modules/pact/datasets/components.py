from typing import *
from abc import abstractmethod
import os
import json
from sympy import root
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2
import copy


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(
        self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(",")
        self.instances = []
        self.metadata = pd.DataFrame()

        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
            self._stats[key]["Total"] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend(
                [(root, sha256) for sha256 in metadata["sha256"].values]
            )
            metadata.set_index("sha256", inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

    @abstractmethod
    def filter_metadata(
        self, metadata: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass

    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Total instances: {len(self)}")
        lines.append(f"  - Sources:")
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for k, v in stats.items():
                lines.append(f"      - {k}: {v}")
        return "\n".join(lines)


import json
from modules.utils.articulation_utils import cat_ref, sem_ref, joint_ref


class StandardDatasetBase_Part(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(
        self,
        roots: str,
        split_info_json: str = None,
        is_test=False,
    ):
        super().__init__()

        self.use_imgs_instance = False
        self.split_info = None
        self.is_test = is_test
        if split_info_json is not None and os.path.exists(split_info_json):
            # self.split_info = json.loads(open(split_info_json))
            with open(split_info_json) as f:
                self.split_info = json.load(f)
            self.split_info = (
                self.split_info["train"] if not is_test else self.split_info["test"]
            )
            print(
                f"Using split info from {split_info_json}, is_test={is_test}, num instances={len(self.split_info)}"
            )
        else:
            print("No split info provided, using all data.")

        self.roots = roots.split(",")
        self.instances = []

        self._stats = {}
        self.metadata = pd.DataFrame()
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}

            metadata = self.prepare_meta(root)

            # metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]["Total"] = len(metadata)
            print("subdataset:", key, " total num:", len(metadata))
            # metadata, stats = self.filter_metadata(metadata)
            if len(metadata) == 0:
                print("No data in root:", root, " has no metadata")
                continue
            self.instances.extend([(root, path) for path in metadata["path"].values])
            metadata.set_index("sha256", inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

    def prepare_meta(
        self,
        path,
    ):

        sub_folder = sorted(os.listdir(path))
        sub_folder = [f for f in sub_folder if os.path.isdir(os.path.join(path, f))]
        metadata = pd.DataFrame(
            columns=["sha256", "path", "n_parts", "scale", "offset"]
        )

        metadata.set_index("sha256", inplace=True)
        for sub in sub_folder:  ## 00000,00001
            # if sub == "47669":
            #     continue
            sub_path = os.path.join(path, sub)
            sub_path = os.path.abspath(sub_path)
            if self.split_info is not None:
                tag1 = "/".join(sub_path.split("/")[-2:])
                tag4 = "/".join(sub_path.split("/")[-4:])
                tag3 = "/".join(sub_path.split("/")[-3:])
                # tag1 = sub_path.split("/")[-2:].join("/")
                # tag2 = sub_path.split("/")[-4:].join("/")
                # cond_1 in
                # print("tag1:", tag1, " tag3:", tag3, " tag4:", tag4)
                if (
                    tag4 not in self.split_info
                    and tag1 not in self.split_info
                    and tag3 not in self.split_info
                ):
                    # if not self.is_test and "hssd-data" in tag2:
                    #     # print(f"Train set skip {tag2}")
                    #     pass
                    # else:
                    continue

            if os.path.isdir(sub_path) and os.path.exists(
                os.path.join(sub_path, "trellis_part_preprocess")
            ):
                with open(os.path.join(sub_path, self.part_info_file)) as f:
                    arti_info = json.load(f)
                with open(
                    os.path.join(
                        sub_path,
                        self.preprocess_folder,
                        self.render_folder,
                        "./full/transforms.json",
                    )
                ) as f:
                    render_info = json.load(f)
                arti_info["scale"] = render_info["scale"]
                arti_info["offset"] = render_info["offset"]

                # for

                # metadata
                ## add to metadata
                new_row = pd.DataFrame(
                    [
                        {
                            "sha256": sub_path,
                            "path": sub_path,
                            "n_parts": len(arti_info["diffuse_tree"]),
                            "scale": arti_info["scale"],
                            "offset": arti_info["offset"],
                        }
                    ]
                )
                metadata = pd.concat([metadata, new_row])
            else:
                print(f"skip {sub_path} as it don't have preprocess folder")
        # metadata.set_index('sha256', inplace=True)

        return metadata

        # metadata.add

    def _prepare_node_data(self, node):
        # semantic label
        label = (
            np.array([sem_ref["fwd"][node["name"]]], dtype=np.float32) / 5.0 - 0.8
        )  # (1,), range from -0.8 to 0.8
        # joint type
        joint_type = (
            np.array([joint_ref["fwd"][node["joint"]["type"]] / 5.0], dtype=np.float32)
            - 0.5
        )  # (1,), range from -0.8 to 0.8
        # aabb
        aabb_max = np.array(node["aabb"]["max_xyz"], dtype=np.float32)
        aabb_min = np.array(node["aabb"]["min_xyz"], dtype=np.float32)
        aabb_center = (aabb_max + aabb_min) / 2.0
        # joint axis and range
        if node["joint"]["type"] == "fixed":
            axis_dir = np.zeros((3,), dtype=np.float32)
            axis_ori = aabb_center
            joint_range = np.zeros((2,), dtype=np.float32)
        else:
            if (
                node["joint"]["type"] == "revolute"
                or node["joint"]["type"] == "continuous"
            ):
                joint_range = (
                    np.array([node["joint"]["range"][1]], dtype=np.float32) / 360.0
                )
                joint_range = np.concatenate(
                    [joint_range, np.zeros((1,), dtype=np.float32)], axis=0
                )  # (2,)
            elif (
                node["joint"]["type"] == "prismatic" or node["joint"]["type"] == "screw"
            ):
                joint_range = np.array([node["joint"]["range"][1]], dtype=np.float32)
                joint_range = np.concatenate(
                    [np.zeros((1,), dtype=np.float32), joint_range], axis=0
                )  # (2,)
            axis_dir = (
                np.array(node["joint"]["axis"]["direction"], dtype=np.float32) * 0.7
            )  # (3,), range from -0.7 to 0.7
            # make sure the axis is pointing to the positive direction
            if np.sum(axis_dir > 0) < np.sum(-axis_dir > 0):
                axis_dir = -axis_dir
                joint_range = -joint_range
            axis_ori = np.array(
                node["joint"]["axis"]["origin"], dtype=np.float32
            )  # (3,), range from -1 to 1
            if (
                node["joint"]["type"] == "prismatic" or node["joint"]["type"] == "screw"
            ) and node["name"] != "door":
                axis_ori = aabb_center
        node_data = np.concatenate(
            [
                joint_type.repeat(6),
                axis_dir,
                axis_ori,
                joint_range.repeat(3),
                label.repeat(6),
            ],
            axis=0,
        )
        mask = np.ones_like(node_data, dtype=np.float32)
        if node["joint"]["type"] == "fixed":
            mask[6:12] = 0  ## mask the joint axis and range for fixed joint
        if self.is_mask_primistic_axis_o and (
            node["joint"]["type"] == "prismatic" or node["joint"]["type"] == "screw"
        ):
            mask[9:12] = 0  ## mask the joint origin for prismatic and screw joint
        return node_data, mask

    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass

    def __len__(self):
        if hasattr(self, "imgs_instances") and self.use_imgs_instance:
            return len(self.imgs_instances)
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            if hasattr(self, "imgs_instances") and self.use_imgs_instance:
                res = self.imgs_instances[index]
            else:
                res = self.instances[index]
            if len(res) == 2:
                root, path = res
                return self.get_instance(path)
            elif len(res) == 3:
                root, path, img_idx = res
                return self.get_instance(path, img_idx)
            return self.get_instance(path)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Total instances: {len(self)}")
        lines.append(f"  - Sources:")
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for k, v in stats.items():
                lines.append(f"      - {k}: {v}")
        return "\n".join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]["captions"])

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata["captions"].notna()]
        stats["With captions"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack["cond"] = text
        return pack


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f"cond_rendered"]]
        stats["Cond rendered"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, "renders_cond", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]

        image_path = os.path.join(image_root, metadata["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [
            center[0] + aug_center_offset[0],
            center[1] + aug_center_offset[1],
        ]
        aug_bbox = [
            int(aug_center[0] - aug_hsize),
            int(aug_center[1] - aug_hsize),
            int(aug_center[0] + aug_hsize),
            int(aug_center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)

        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack["cond"] = image

        return pack


import einops
from modules.inference_utils import load_img_mask


class ImageConditionedMixin_PartNet:
    def __init__(self, roots, *, image_size=518, img_cond_cache=False, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
        if img_cond_cache:
            self.data_cache_cond = {}
        self.is_sorted_mask_bottom_up = kwargs.get("is_sorted_mask_bottom_up", True)
        print(
            f"ImageConditionedMixin_PartNet: is_sorted_mask_bottom_up={self.is_sorted_mask_bottom_up}"
        )

        self.is_ablation_mask = kwargs.get("is_ablation_mask", False)
        # self.img_cond_cache = img_cond_cache
        self.use_imgs_instance = kwargs.get("use_imgs_instance", False)

        print(f"use_imgs_instance: {self.use_imgs_instance}")

        self.imgs_instances = []
        if self.is_test:
            img_idxs = list(range(18, 20))
        else:
            img_idxs = list(range(0, 18))
        self.img_idxs = img_idxs
        for instance in self.instances:
            root, path = instance
            for img_idx in self.img_idxs:
                self.imgs_instances.append((root, path, img_idx))

    # def filter_metadata(self, metadata):
    #     metadata, stats = super().filter_metadata(metadata)
    #     metadata = metadata[metadata[f'cond_rendered']]
    #     stats['Cond rendered'] = len(metadata)
    #     return metadata, stats

    def get_instance(self, instance_path, img_idx=None):
        pack = super().get_instance(instance_path)
        if img_idx is not None:
            idx = str(img_idx).zfill(2)
            pack = copy.deepcopy(pack)
        else:
            if not self.is_test:

                num_cond = 18
                idx = np.random.randint(num_cond)
            elif self.is_test:
                num_cond = 2
                idx = np.random.randint(num_cond)
                idx = 18 + idx
            ## idx
            idx = str(idx).zfill(2)
        if (
            hasattr(self, "data_cache_cond")
            and instance_path in self.data_cache_cond
            and idx in self.data_cache_cond[instance_path]
        ):
            pack_cond = self.data_cache_cond[instance_path][idx]

        else:
            if hasattr(self, "data_cache_cond"):
                if instance_path not in self.data_cache_cond:
                    self.data_cache_cond[instance_path] = {}

            part_mask_dir = os.path.join(
                instance_path, f"{self.msk_cond_folder}/{idx}.npz"
            )
            rgba_img_dir = os.path.join(
                instance_path, f"{self.img_cond_folder}/{idx}.png"
            )

            msk_semantic = np.load(part_mask_dir)
            img_rgb = Image.open(rgba_img_dir)
            mask_semantic = msk_semantic.get("semantic_mask", None).astype(np.int32)

            alpha_msk = np.array(img_rgb.getchannel(3)) / 255.0 > 0.5

            mask_semantic = cv2.resize(
                mask_semantic,
                (alpha_msk.shape[1], alpha_msk.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            wh = (self.image_size, self.image_size)
            mask_semantic[alpha_msk] = mask_semantic[alpha_msk] + 1
            img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = (
                load_img_mask(
                    None,
                    None,
                    size=wh,
                    img=img_rgb,
                    mask_part=einops.repeat(mask_semantic, "h w -> h w c", c=1),
                    is_sort=self.is_sorted_mask_bottom_up,
                )
            )
            if self.is_ablation_mask:
                ordered_mask_input = torch.zeros_like(ordered_mask_input)

            pack_cond = {
                "id": instance_path + f"_{idx}",
                "cond": img_black_bg,
                "masks": ordered_mask_input,
                "ordered_mask_dino": ordered_mask_input,
                "img_mask_vis": torch.tensor(np.asarray(img_mask_vis))
                .permute(2, 0, 1)
                .float()[:, :, :]
                / 255.0,
            }

            if hasattr(self, "data_cache_cond"):
                self.data_cache_cond[instance_path][idx] = pack_cond

        pack.update(pack_cond)
        return pack


import imageio.v3 as iio
import numpy as np


class ImageConditioned_dataset:
    def __init__(self, roots, *, image_size=518, is_depth_one_dir=False, **kwargs):
        # super().__init__(roots, image_size=image_size, img_cond_cache=img_cond_cache,  **kwargs)
        super().__init__()

        self.image_size = image_size
        self.roots = roots.split(",")
        from glob import glob

        print("Using ImageConditioned_Datatset")
        self.instance_paths = []
        for root in self.roots:
            key = os.path.basename(root)
            sub_folders = sorted(os.listdir(root))
            sub_folders = sorted(
                [f for f in sub_folders if os.path.isdir(os.path.join(root, f))]
            )

            if is_depth_one_dir:
                sub_folders = [""]

            for sub in sub_folders:
                sub_path = os.path.join(root, sub)
                sub_path = os.path.abspath(sub_path)
                print(sub_path)
                if (
                    os.path.isdir(sub_path)
                    and len(glob(os.path.join(sub_path, "*_processed.png"))) > 0
                ):
                    print(sub_path)
                    for img_path in sorted(
                        glob(os.path.join(sub_path, "*_processed.png"))
                    ):
                        mask_path = img_path.replace("_processed.png", "_mask.exr")
                        if os.path.exists(mask_path):
                            self.instance_paths.append((img_path, mask_path))
        self.is_sorted_mask_bottom_up = kwargs.get("is_sorted_mask_bottom_up", True)

    def __getitem__(self, index) -> Dict[str, Any]:
        instance_path = self.instance_paths[index]
        img_path = instance_path[0]
        mask_path = instance_path[1]

        mask = iio.imread(mask_path)
        img_rgb = Image.open(img_path)

        mask_semantic = mask[..., 0]
        alpha_msk = np.array(img_rgb.getchannel(3)) / 255.0 > 0.5
        wh = (self.image_size, self.image_size)
        mask_semantic[alpha_msk] = mask_semantic[
            alpha_msk
        ]  ###  NOTE: mask here starts from 0, and 0 is also the valid part id for background, so we don't add 1 here
        mask_semantic = mask_semantic.astype(np.int32)
        img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(
            None,
            None,
            size=wh,
            img=img_rgb,
            mask_part=einops.repeat(mask_semantic, "h w -> h w c", c=1),
            is_sort=True,
        )
        _, _, unordered_mask_input, _ = load_img_mask(
            None,
            None,
            size=wh,
            img=img_rgb,
            mask_part=einops.repeat(mask_semantic, "h w -> h w c", c=1),
            is_sort=False,
        )

        pack_cond = {
            "id": img_path,
            "cond": img_black_bg,
            "masks": ordered_mask_input,
            "unordered_masks": unordered_mask_input,
            "ordered_mask_dino": ordered_mask_input,
            "img_mask_vis": torch.tensor(np.asarray(img_mask_vis))
            .permute(2, 0, 1)
            .float()[:, :, :]
            / 255.0,
            "num_parts": int(ordered_mask_input.max().item()),
            "part_idx": torch.tensor(
                list(range(0, int(ordered_mask_input.max().item())))
            ).to(torch.int32),
        }
        return pack_cond

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]  #  [[0, 1]， [2, 3]]

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            latents = []
            num_parts = []
            part_idx = []
            start = 0
            start_art = 0
            for i, b in enumerate(sub_batch):
                num_parts_i = b["num_parts"]
                num_parts.append(num_parts_i)
                part_idx.append(b["part_idx"])

            pack["num_parts"] = torch.tensor(num_parts).to(torch.int32)
            # collate sparse tensor
            part_idx = torch.cat(part_idx)
            pack["part_idx"] = part_idx
            keys = [
                k
                for k in sub_batch[0].keys()
                if k
                not in [
                    "x_0",
                    "part_idx",
                    "num_parts",
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
                    # assert pack[k].shape[0] == pack['x_0'].shape[0], f"collated {k} shape[0] should be equal to x_0 shape[0]! {pack[k].shape[0]}, {pack['x_0'].shape[0]}"
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

    def __len__(self):
        return len(self.instance_paths)
