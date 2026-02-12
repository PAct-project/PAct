import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import (
    StandardDatasetBase,
    TextConditionedMixin,
    ImageConditionedMixin,
    # ImageConditionedMixin_PartNet,
    StandardDatasetBase_Part,
)
from .sparse_structure import SparseStructure_PartNet
from ..modules.sparse.basic import SparseTensor, sparse_cat
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.data_utils import load_balanced_group_indices
import trimesh
from ..process_utils import merge_gaussians, exploded_gaussians
from modules.utils.articulation_utils import (
    cat_ref,
    sem_ref,
    joint_ref,
    animate_and_render,
)
from modules.pact.utils import render_utils, postprocessing_utils
import pickle
from modules.utils.articulation_utils import _apply_transform_gaussian


def filter_duplicate_coords(noise_coords, existing_coords):
    """Return rows from noise_coords that are not present in existing_coords.

    Both inputs are expected to be array-like with shape (N, D). The comparison
    is exact (no tolerance) and optimized for integer arrays using a void-view
    trick. Falls back to tuple/set checking when necessary.

    Args:
        noise_coords: array-like of shape (M, D)
        existing_coords: array-like of shape (N, D)

    Returns:
        np.ndarray: filtered noise_coords containing only rows not present in existing_coords.
    """
    noise_coords = np.asarray(noise_coords)
    existing_coords = np.asarray(existing_coords)
    # fast exits
    if noise_coords.size == 0:
        return noise_coords
    if existing_coords.size == 0:
        return noise_coords

    try:
        a = np.ascontiguousarray(existing_coords)
        b = np.ascontiguousarray(noise_coords)
        if a.dtype == b.dtype and a.shape[1] == b.shape[1]:
            void_dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            a_view = a.view(void_dtype).ravel()
            b_view = b.view(void_dtype).ravel()
            keep_mask = ~np.in1d(b_view, a_view)
            if keep_mask.sum() == 0:
                return np.zeros(
                    (0, existing_coords.shape[1]), dtype=existing_coords.dtype
                )
            return noise_coords[keep_mask]
        else:
            existing_set = set(map(tuple, existing_coords))
            keep_mask = np.array(
                [tuple(row) not in existing_set for row in noise_coords], dtype=bool
            )
            if keep_mask.sum() == 0:
                return np.zeros(
                    (0, existing_coords.shape[1]), dtype=existing_coords.dtype
                )
            return noise_coords[keep_mask]
    except Exception:
        # best-effort fallback to tuple-based filtering
        try:
            existing_set = set(map(tuple, existing_coords))
            keep_mask = np.array(
                [tuple(row) not in existing_set for row in noise_coords], dtype=bool
            )
            if keep_mask.sum() == 0:
                return np.zeros(
                    (0, existing_coords.shape[1]), dtype=existing_coords.dtype
                )
            return noise_coords[keep_mask]
        except Exception:
            # give up and return original noise_coords unchanged
            return noise_coords


def clear_empty_part(reps_parts, arti_info, textured_mesh_parts=None, gs_parts=None):
    assert len(reps_parts) == len(arti_info)
    if textured_mesh_parts is not None:
        assert len(reps_parts) == len(textured_mesh_parts)
    if gs_parts is not None:
        assert len(reps_parts) == len(gs_parts)
    Non_empty_index = []
    for idx_, arti_part_info in enumerate(arti_info):
        rep_part = reps_parts[idx_]
        rep_part_vertexs = rep_part.vertices.cpu().numpy()
        Non_empty_index.append(idx_) if rep_part_vertexs.shape[0] > 0 else None
    if len(Non_empty_index) == len(arti_info):

        return reps_parts, arti_info, textured_mesh_parts, gs_parts
    ## print in red alert
    print(
        "\033[91m[Warning] Some empty parts are found and removed during exporting to Singapo style.",
        len(arti_info) - len(Non_empty_index),
        Non_empty_index,
        "\033[0m",
    )

    reps_parts = [reps_parts[idx_] for idx_ in Non_empty_index]
    arti_info = [arti_info[idx_] for idx_ in Non_empty_index]
    if textured_mesh_parts is not None:
        textured_mesh_parts = [textured_mesh_parts[idx_] for idx_ in Non_empty_index]
    if gs_parts is not None:
        gs_parts = [gs_parts[idx_] for idx_ in Non_empty_index]

    ### re - sort index from 0 to N-1
    for idx, arti_part_info in enumerate(arti_info):
        arti_part_info["id"] = idx
    return reps_parts, arti_info, textured_mesh_parts, gs_parts


def export_arti_obj_to_singapo_style(
    reps_parts,
    arti_info,
    export_dir="./outputs/tmp/singapo_export",
    transfor_matrix=None,
    scale_shift=None,
    textured_mesh_parts=None,
    gs_parts=None,
):
    if scale_shift is None:
        scale = 1.0
        translation = np.array([0.0, 0.0, 0.0])
    else:
        scale, translation = scale_shift
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    if len(arti_info) != len(reps_parts):
        raise ValueError(
            f"Length of arti_info ({len(arti_info)}) does not match length of reps_parts ({len(reps_parts)})"
        )

    reps_parts, arti_info, textured_mesh_parts, gs_parts = clear_empty_part(
        reps_parts, arti_info, textured_mesh_parts, gs_parts
    )
    ##

    root_idx = [
        arti_part_info["id"]
        for arti_part_info in arti_info
        if (
            arti_part_info["name"] == "base"
            and arti_part_info["joint"]["type"] == "fixed"
        )
    ]
    if len(root_idx) == 1:
        root_idx = root_idx[0]
    else:
        largest_size_id = -1
        prev_size = -1
        for idx_, arti_part_info in enumerate(arti_info):
            if arti_part_info["id"] in root_idx:
                rep_part = reps_parts[idx_]
                rep_part_vertexs = rep_part.vertices.cpu().numpy()
                try:
                    aabb_min = rep_part_vertexs.min(axis=0)
                    aabb_max = rep_part_vertexs.max(axis=0)
                    size = np.prod(aabb_max - aabb_min)
                    if size > prev_size:
                        prev_size = size
                        largest_size_id = arti_part_info["id"]
                except Exception as e:
                    print("Error in computing aabb for part:", arti_part_info["id"], e)
        root_idx = largest_size_id
    other_idx = [
        arti_part_info["id"]
        for arti_part_info in arti_info
        if arti_part_info["id"] != root_idx
    ]

    # raise ValueError("There should be exactly one root part named 'base' with fixed joint.")
    for arti_part_info in arti_info:
        node_id = arti_part_info.get("id")
        if root_idx == node_id:
            arti_part_info["parent"] = -1
            arti_part_info["joint"]["type"] = "fixed"
            arti_part_info["children"] = other_idx
            # arti_part_in
            continue
        else:
            arti_part_info["parent"] = root_idx
            arti_part_info["children"] = []

    # arti_info = sorted(arti_info, key=lambda x: x['id'])

    for idx, (arti_part_info, part_rep) in enumerate(zip(arti_info, reps_parts)):
        vertexs = part_rep.vertices.cpu().numpy()  ###
        vertexs = (
            1 / scale * (vertexs - translation)
        )  ### from TRELLIS blender space to PartNet_mobility space
        part_rep = trimesh.Trimesh(
            vertices=vertexs,
            faces=part_rep.faces.cpu().numpy(),
            face_normal=part_rep.face_normal.cpu().numpy(),
            vertex_attrs=part_rep.vertex_attrs.cpu().numpy(),
        )
        if textured_mesh_parts:
            texture_mesh = textured_mesh_parts[idx]
            vertexs_tex = texture_mesh.vertices
            vertexs_tex = 1 / scale * (vertexs_tex - translation)
            textured_mesh_parts[idx].vertices = vertexs_tex
        if gs_parts:
            gs_part = gs_parts[idx]
            centers = gs_part._xyz
            centers = (
                1 / scale * (centers - torch.Tensor(translation).to(centers.device))
            )
            gs_part._xyz = centers

        if transfor_matrix is not None:
            ### transform axis_o, axis_d, center, aabb
            arti_part_info["joint"]["axis"]["origin"] = (
                1
                / scale
                * (np.array(arti_part_info["joint"]["axis"]["origin"]) - translation)
                @ transfor_matrix[:3, :3].T
            ).tolist()
            arti_part_info["joint"]["axis"]["direction"] = (
                np.array(arti_part_info["joint"]["axis"]["direction"])
                @ transfor_matrix[:3, :3].T
            ).tolist()
            part_rep.apply_transform(transfor_matrix)
            if textured_mesh_parts:
                texture_mesh = textured_mesh_parts[idx]
                texture_mesh.apply_transform(transfor_matrix)
            if gs_parts:
                gs_part = gs_parts[idx]
                gs_parts[idx] = _apply_transform_gaussian(gs_part, transfor_matrix)

        out_path = f"ply/part_{arti_part_info['id']}.ply"
        export_ply_path = os.path.join(export_dir, out_path)
        if not os.path.exists(os.path.dirname(export_ply_path)):
            os.makedirs(os.path.dirname(export_ply_path))
        ## TODO: Transform from different coords system if needed.
        # postprocessing_utils.export_mesh_to_ply(part_rep, export_ply_path)
        part_rep.export(export_ply_path)
        arti_part_info["plys"] = [
            out_path,
        ]
        vertexs = part_rep.vertices
        try:  ### update transformed aabb
            aabb_min = vertexs.min(axis=0)
            aabb_max = vertexs.max(axis=0)
            arti_part_info["aabb"]["center"] = ((aabb_max + aabb_min) / 2).tolist()
            arti_part_info["aabb"]["size"] = (aabb_max - aabb_min).tolist()
        except Exception as e:
            print("Error in computing aabb for part:", arti_part_info["id"], e)
            arti_part_info["aabb"]["center"] = [0.0, 0.0, 0.0]
            arti_part_info["aabb"]["size"] = [0.0, 0.0, 0.0]

        if textured_mesh_parts:
            texture_mesh = textured_mesh_parts[idx]
            texture_export_path = os.path.join(
                export_dir, f"glb/part_{arti_part_info['id']}.glb"
            )
            if not os.path.exists(os.path.dirname(texture_export_path)):
                os.makedirs(os.path.dirname(texture_export_path))
            texture_mesh.export(texture_export_path)
            arti_part_info["glb"] = [
                f"glb/part_{arti_part_info['id']}.glb",
            ]
        if gs_parts:
            gs_part = gs_parts[idx]
            gs_export_path = os.path.join(
                export_dir, f"gs/part_{arti_part_info['id']}.ply"
            )
            if not os.path.exists(os.path.dirname(gs_export_path)):
                os.makedirs(os.path.dirname(gs_export_path))

            gs_part.save_ply(gs_export_path)
            arti_part_info["gs"] = [
                f"gs/part_{arti_part_info['id']}.ply",
            ]

    ### export articulation json
    export_json_path = os.path.join(export_dir, "object.json")

    to_dump = {"diffuse_tree": arti_info}
    with open(export_json_path, "w") as f:
        json.dump(to_dump, f, indent=4)


class SLatVisMixin:
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        self.vis_video = self.kwargs.get("vis_video", False)

    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(
                **cfg["models"]["decoder"]["args"]
            )
            ckpt_path = os.path.join(
                self.slat_dec_path, "ckpts", f"decoder_{self.slat_dec_ckpt}.pt"
            )
            decoder.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True)
            )
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.slat_dec
        self.slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            reps.append(self.slat_dec(z[i : i + batch_size]))
        reps = sum(reps, [])
        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0_info: Union[SparseTensor, dict]):
        x_0 = x_0_info if isinstance(x_0_info, SparseTensor) else x_0_info["x_0"]
        if "part_layouts" in x_0_info:
            part_layouts = x_0_info["part_layouts"]

            reps = self.decode_latent(x_0.cuda(), part_layouts=part_layouts)
        else:
            reps = self.decode_latent(x_0.cuda())

        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
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
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:  ### list of [gs ,gs,gs ...]
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)
        images = torch.stack(images)
        video = None
        if self.vis_video:
            videos = []
            for i, representation in enumerate(reps):
                video = render_utils.render_video(
                    representation,
                    num_frames=60,
                    bg_color=(1, 1, 1),
                    r=2.3,
                    fov=60,
                    verbose=False,
                )["color"]
                videos.append(video)

        return images, videos


from modules.utils.articulation_utils import (
    cat_ref,
    sem_ref,
    joint_ref,
    convert_json,
    convert_data_range,
    convert_data_2_info,
    post_process_arti_info,
)


class Arti_SLatVisMixin(SLatVisMixin):
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
        pretrained_slat_dec_mesh: str = "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_slat_dec_mesh = pretrained_slat_dec_mesh

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

    @torch.no_grad()
    def visualize_sample(self, x_0_info: Union[SparseTensor, dict]):
        x_0 = x_0_info if isinstance(x_0_info, SparseTensor) else x_0_info["x_0"]

        if "part_layouts" in x_0_info:
            part_layouts = x_0_info["part_layouts"]

            reps, reps_part = self.decode_latent(x_0.cuda(), part_layouts=part_layouts)
        else:
            reps = self.decode_latent(x_0.cuda())

        if "x_0_arti" in x_0_info:
            videos_arti = []
            # for key_suffix in ['_gt','']:
            # key = "x_0_arti"+key_suffix
            key = "x_0_arti"
            x_0_arti = x_0_info[key]

            x_0_arti_list = []
            for idx in range(x_0_arti.shape[0]):
                num_art_c = 24
                arti_info = convert_data_2_info(
                    x_0_arti[idx].feats[:, :num_art_c].cpu().numpy()
                )
                x_0_arti_list.append(arti_info)
            videos_arti = self.animate_with_articulation(reps_part, x_0_arti_list)
            # self.visualize_arti_info(x_0_arti)
            # x_0_arti = x_0_info["x_0_arti_gt"]
            # x_0_arti_list =[]
            # for idx in range(x_0_arti.shape[0]):
            #     num_art_c = 24
            #     arti_info = convert_data_2_info(x_0_arti[idx].feats[:,:num_art_c].cpu().numpy())
            #     x_0_arti_list.append(arti_info)
            # video_arti_samples = self.animate_with_articulation(reps_part, x_0_arti_list)

        else:
            videos_arti = None
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
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
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:  ### list of [gs ,gs,gs ...]
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)
        images = torch.stack(images)
        video = None
        if self.vis_video:
            videos = []
            for i, representation in enumerate(reps):
                video = render_utils.render_video(
                    representation,
                    num_frames=60,
                    bg_color=(1, 1, 1),
                    r=2.3,
                    fov=60,
                    verbose=False,
                )["color"]
                videos.append(video)

        return images, videos, videos_arti

    @torch.no_grad()
    def decode_latent(self, z, part_layouts=None, batch_size=4):
        """decoding into 3DGS for visualization"""
        self._loading_slat_dec()
        reps_merged = []
        reps_parts = []
        assert z.shape[0] == len(
            part_layouts
        ), f"z.shape[0] ({z.shape[0]}) should be equal to len(part_layouts) ({len(part_layouts)})"
        if self.normalization is not None:
            z.feats[:, :8] = z.feats[:, :8] * self.std.to(z.device) + self.mean.to(
                z.device
            )
        for i in range(0, z.shape[0], batch_size):
            res = self.slat_dec(
                self.divide_slat(
                    z[i : i + batch_size], part_layouts[i : i + batch_size]
                )
            )

            ### reorganize the output according to part_layouts
            start = 0
            for part_layout in part_layouts[i : i + batch_size]:

                exploded_gs = exploded_gaussians(
                    res[start + 1 : start + len(part_layout)], explosion_scale=0.3
                )
                # exploded_gs.save_ply(f"{output_dir}/exploded_gs.ply")
                reps_merged.append(exploded_gs)
                reps_parts.append(res[start + 1 : start + len(part_layout)])
                start += len(part_layout)
            # reps.append(self.slat_dec(self.divide_slat(z[i:i+batch_size], part_layouts[i:i+batch_size])))

        # reps = sum(reps, [])
        self._delete_slat_dec()
        return reps_merged, reps_parts

    @torch.no_grad()
    def decode_latent_mesh(self, z, part_layouts=None, batch_size=1):
        """decoding into mesh for visualization"""
        temp_docoder_dir = self.pretrained_slat_dec
        self.pretrained_slat_dec = (
            self.pretrained_slat_dec_mesh
        )  ## switch to mesh decoder
        self._loading_slat_dec()
        self.pretrained_slat_dec = temp_docoder_dir  ## restore the decoder path

        reps_parts = []
        assert z.shape[0] == len(
            part_layouts
        ), f"z.shape[0] ({z.shape[0]}) should be equal to len(part_layouts) ({len(part_layouts)})"
        if self.normalization is not None:
            z.feats[:, :8] = z.feats[:, :8] * self.std.to(z.device) + self.mean.to(
                z.device
            )
        for i in range(0, z.shape[0], batch_size):
            res = self.slat_dec(
                self.divide_slat(
                    z[i : i + batch_size], part_layouts[i : i + batch_size]
                )
            )
            start = 0
            for part_layout in part_layouts[i : i + batch_size]:

                reps_parts.append(res[start + 1 : start + len(part_layout)])
                start += len(part_layout)

        self._delete_slat_dec()
        return reps_parts

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
                    coords=slat[part_id].coords[part_slice],  ### NOTE: :TODO
                    feats=slat[part_id].feats[part_slice],
                )
                sparse_part.append(part_x_sparse_tensor)

        slat = sparse_cat(sparse_part)

        return self.remove_noise(slat)

    def remove_noise(self, z_batch):
        """
        Remove noise from latent vectors by filtering out points with low confidence.

        Args:
            z_batch: Latent vectors to process

        Returns:
            sp.SparseTensor: Processed latent with noise removed
        """
        # Create a new list for processed tensors
        processed_batch = []

        for i, z in enumerate(z_batch):
            coords = z.coords
            feats = z.feats

            # Only filter if features have a confidence dimension (9th dimension)
            if feats.shape[1] == 9:
                # Get the confidence values (last dimension)
                last_dim = feats[:, -1]
                sigmoid_val = torch.sigmoid(last_dim)

                # Calculate filtering statistics
                total_points = coords.shape[0]
                to_keep = sigmoid_val >= 0.5
                kept_points = to_keep.sum().item()
                discarded_points = total_points - kept_points
                discard_percentage = (
                    (discarded_points / total_points) * 100 if total_points > 0 else 0
                )

                if kept_points == 0:
                    if self.verbose:
                        print(f"No points kept for part {i}")
                    continue

                if self.verbose:
                    print(
                        f"Discarded {discarded_points}/{total_points} points ({discard_percentage:.2f}%)"
                    )

                # Filter coordinates and features
                coords = coords[to_keep]
                feats = feats[to_keep]
                feats = feats[:, :-1]  # Remove the confidence dimension

                # Create a filtered SparseTensor
                processed_z = z.replace(coords=coords, feats=feats)
            else:
                processed_z = z

            processed_batch.append(processed_z)

        return sparse_cat(processed_batch)


class SLat(SLatVisMixin, StandardDatasetBase):
    """
    structured latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        self.loads = [
            self.metadata.loc[sha256, "num_voxels"] for _, sha256 in self.instances
        ]

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(1, -1)
            self.std = torch.tensor(self.normalization["std"]).reshape(1, -1)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f"latent_{self.latent_model}"]]
        stats["With latent"] = len(metadata)
        metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
        stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        metadata = metadata[metadata["num_voxels"] <= self.max_num_voxels]
        stats[f"Num voxels <= {self.max_num_voxels}"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        data = np.load(
            os.path.join(root, "latents", self.latent_model, f"{instance}.npz")
        )
        coords = torch.tensor(data["coords"]).int()
        feats = torch.tensor(data["feats"]).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            "coords": coords,
            "feats": feats,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b["coords"].shape[0] for b in batch], split_size
            )
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(
                    torch.cat(
                        [
                            torch.full((b["coords"].shape[0], 1), i, dtype=torch.int32),
                            b["coords"],
                        ],
                        dim=-1,
                    )
                )
                feats.append(b["feats"])
                layout.append(slice(start, start + b["coords"].shape[0]))
                start += b["coords"].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack["x_0"] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack["x_0"]._shape = torch.Size(
                [len(group), *sub_batch[0]["feats"].shape[1:]]
            )
            pack["x_0"].register_spatial_cache("layout", layout)

            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ["coords", "feats"]]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


from modules.utils.part_utils import (
    merge_duplicated_voxels,
    change_bbox_range,
    merge_multipart_slat,
)
from modules.utils.part_utils import (
    gen_mesh_from_bounds_no_transform,
    extract_bbox_info,
    convert_to_axis_aligned_bbox,
    load_meshes,
    build_aabb_array,
    sort_meta_by_bboxes_zyx,
    sort_bboxes_by_zyx,
    prepaer_parts_meta,
)
from modules.inference_utils import load_img_mask
from tqdm import tqdm
import os
import concurrent.futures


class Arti_SLat(Arti_SLatVisMixin, StandardDatasetBase_Part):
    """
    structured latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
        pretrained_slat_dec_mesh: str = "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        verbose: bool = False,
        split_info_json: Optional[str] = None,
        is_test: bool = False,
        **kwargs,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        self.verbose = verbose
        self.kwargs = kwargs
        self.preset_alpha = self.kwargs.get("preset_alpha", 6)
        self.render_folder = self.kwargs.get(
            "render_folder", "render_merged_fixed_eevee"
        )
        self.latent_folder = self.kwargs.get("latent_folder", "latents")
        self.latent_model_name = self.kwargs.get(
            "latent_model_name", "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
        )
        self.voxel_folder = self.kwargs.get("voxel_folder", "voxels_merged_fixed")
        self.img_cond_folder = self.kwargs.get("img_cond_folder", "imgs")
        self.msk_cond_folder = self.kwargs.get(
            "msk_cond_folder", "imgs/semantic_masks_merge_fixed"
        )
        self.preprocess_folder = self.kwargs.get(
            "preprocess_folder", "trellis_part_preprocess"
        )
        self.part_info_file = (
            "object_merge_fixed.json"
            if self.kwargs.get("merge_fixed_part", True)
            else "object.json"
        )
        self.transform_info_file = os.path.join(
            "trellis_part_preprocess", self.render_folder, "full/transforms.json"
        )
        self.is_predict_arti_info = self.kwargs.get(
            "is_predict_arti_info", False
        )  ### TODO
        self.adding_slat_noise = self.kwargs.get("adding_slat_noise", True)  ### TODO
        self.is_mask_primistic_axis_o = self.kwargs.get(
            "is_mask_primistic_axis_o", False
        )  ### TODO
        ## print config in red color
        print("\033[91m" + "Arti_SLat config:" + "\033[0m")

        print("adding_slat_noise:", self.adding_slat_noise)
        print("is_mask_primistic_axis_o:", self.is_mask_primistic_axis_o)
        super().__init__(
            roots,
            split_info_json=split_info_json,
            is_test=is_test,
            pretrained_slat_dec=pretrained_slat_dec,
            pretrained_slat_dec_mesh=pretrained_slat_dec_mesh,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        self.data_cache_slat_part = {}

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(1, -1)
            self.std = torch.tensor(self.normalization["std"]).reshape(1, -1)

        paths = [path for _, path in self.instances]
        n_workers = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(
                tqdm(
                    executor.map(lambda p: Arti_SLat.get_instance(self, p), paths),
                    total=len(paths),
                    desc=f"Preloading instances (workers={n_workers})",
                )
            )
        ###  loads
        self.loads = [
            (
                self.data_cache_slat_part[instant_path]["coords"].shape[0]
                if self.data_cache_slat_part[instant_path]
                else 0
            )
            for _, instant_path in self.instances
        ]
        ## list of np.int64
        print("max loads", max(self.loads), "\n", "min load", min(self.loads))

        ################# Check dataset integrity #################
        ################# Check dataset integrity #################
        ################# Check dataset integrity #################
        def check_folder_exist(instance_path):
            if not os.path.exists(instance_path):
                raise ValueError(f"Folder {instance_path} does not exist.")

            for idx in range(20):
                idx = str(idx).zfill(2)
                if not os.path.exists(instance_path):
                    raise ValueError(
                        f"File {instance_path} does not exist in folder {instance_path}."
                    )

                part_mask_dir = os.path.join(
                    instance_path, f"{self.msk_cond_folder}/{idx}.npz"
                )
                rgba_img_dir = os.path.join(
                    instance_path, f"{self.img_cond_folder}/{idx}.png"
                )
                if not os.path.exists(part_mask_dir):
                    # raise ValueError(f"File {part_mask_dir} does not exist in folder {instance_path}.")
                    print(
                        f"File {part_mask_dir} does not exist in folder {instance_path}."
                    )
                if not os.path.exists(rgba_img_dir):
                    # raise ValueError(f"File {rgba_img_dir} does not exist in folder {instance_path}.")
                    print(
                        f"File {rgba_img_dir} does not exist in folder {instance_path}."
                    )
                pass

        paths = [path for _, path in self.instances]
        n_workers = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(
                tqdm(
                    executor.map(lambda p: check_folder_exist(p), paths),
                    total=len(paths),
                    desc=f"Check dataset sanity (workers={n_workers})",
                )
            )

        ################# Check dataset integrity #################
        ################# Check dataset integrity #################
        ################# Check dataset integrity #################

        if not is_test:
            ## filter out instances with loads > 100000 for training, but keep them for testing
            filtered_obj = []
            for idx, (root, path) in enumerate(self.instances):
                if self.loads[idx] > 100000:
                    pass
                else:
                    filtered_obj.append((root, path))
            self.instances = filtered_obj
            print("[Training Mode] After filtered", len(self.instances))
        self.loads = [
            (
                self.data_cache_slat_part[instant_path]["coords"].shape[0]
                if self.data_cache_slat_part[instant_path]
                else 0
            )
            for _, instant_path in self.instances
        ]

    def init_loads(self):

        self.loads = [
            Arti_SLat.get_instance(self, path) for sha256, path in self.instances
        ]

    def merge_parts(self, latent_list):
        pass

    def make_omnipart_training_slat(self, object_path, part_info_sorted=None):
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

            latent_list.append(
                {
                    "suffix": part_suffix,
                    "coords": part_latent["coords"],
                    "feats": np.concatenate(
                        [
                            part_latent["feats"],
                            self.preset_alpha
                            * np.ones_like(part_latent["feats"][:, :1]),
                        ],
                        axis=-1,
                    ),
                }
            )

        ### add noise voxel in the same bbox:
        assert len(part_info_sorted) == len(
            latent_list
        ), f"part_info_sorted and latent_list should have the same length! {len(part_info_sorted)}, {len(latent_list)}"
        new_latent_list = []

        # Compute other-parts' voxels inside each part's bbox
        noise_coords_list = []
        for i, (part, latent) in enumerate(zip(part_info_sorted, latent_list)):
            bbox_min = np.array(part["aabb"]["min_xyz"])
            bbox_max = np.array(part["aabb"]["max_xyz"])

            # collect coords from other parts that lie inside this part's bbox
            collected = []
            for j, other in enumerate(latent_list):
                if j == i:
                    continue
                coords_other = np.asarray(other["coords"])
                if coords_other.size == 0:
                    continue
                # ensure coords_other has shape (N, >=3), use first 3 columns as xyz
                coords_xyz = coords_other[:, :3]

                bbox = change_bbox_range(
                    np.stack([bbox_min, bbox_max], axis=0), bins=64
                )
                # broadcast compare
                bbox_min, bbox_max = bbox[0,], bbox[1,]

                inside_mask = np.all(
                    (coords_xyz > bbox_min) & (coords_xyz < bbox_max), axis=1
                )
                if inside_mask.any():
                    collected.append(coords_other[inside_mask])
            if len(collected) > 0 and self.adding_slat_noise:
                noise_coords_i = np.concatenate(collected, axis=0)
            else:
                # empty array with shape (0, coords_dim)
                noise_coords_i = np.zeros(
                    (0, latent["coords"].shape[1]), dtype=latent["coords"].dtype
                )
            # Filter out duplicate coordinates that already exist in this part
            noise_coords_i = filter_duplicate_coords(noise_coords_i, latent["coords"])
            noise_coords_list.append(noise_coords_i)
            if self.verbose:

                print(
                    f"part {i} ({part['name']}): {latent['coords'].shape[0]} found another {noise_coords_i.shape[0]} voxels from other parts inside its bbox"
                )

        # Attach other_coords into latent_list entries so downstream code can use them if needed
        for i, (latent, coords_noise) in enumerate(zip(latent_list, noise_coords_list)):
            # latent["other_coords"] = other_coords
            coords_new = np.concatenate([latent["coords"], coords_noise], axis=0)
            feats_noise = np.zeros(
                (coords_noise.shape[0], latent["feats"].shape[1]),
                dtype=latent["feats"].dtype,
            )
            feats_noise[:, -1] = -1 * self.preset_alpha  ## trellis --> omnipart

            feats_new = np.concatenate([latent["feats"], feats_noise], axis=0)
            latent = {
                "suffix": latent["suffix"],
                "coords": coords_new,
                "feats": feats_new,
            }

            new_latent_list.append(latent)

        merged_slat = merge_multipart_slat(new_latent_list, verbose=self.verbose)

        return merged_slat

    def make_omnipart_slat(
        self, object_path, part_info_sorted=None, is_full_body_Mode=None
    ):
        base_dir = object_path
        if part_info_sorted is None:
            pass
            raise NotImplementedError

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

            latent_list.append(
                {
                    "coords": part_latent["coords"],
                    "feats": np.concatenate(
                        [
                            part_latent["feats"],
                            self.preset_alpha
                            * np.ones_like(part_latent["feats"][:, :1]),
                        ],
                        axis=-1,
                    ),
                }
            )
        merged_slat = merge_multipart_slat(latent_list)

        return merged_slat

    def make_omnipart_slat(self, path):
        ### load_part:

        ### load_config

        ### sort bbox by zyx

        ## aggregate all parts

        pass

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

    def _recover_node_data(
        self,
        node_array: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Recover articulation metadata from the flattened node array.

        Args:
            node_array: Array of length 24 produced by ``_prepare_node_data``.
            mask: Optional mask array (same shape) describing valid entries.

        Returns:
            dict: Reconstructed articulation information with joint semantics.
        """

        if isinstance(node_array, torch.Tensor):
            node_array = node_array.detach().cpu().numpy()
        node_array = np.asarray(node_array, dtype=np.float32).flatten()
        if node_array.size != 24:
            raise ValueError(f"Expected node_array of length 24, got {node_array.size}")

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask, dtype=np.float32).flatten()
            if mask.size != 24:
                raise ValueError(f"Expected mask of length 24, got {mask.size}")

        # joint type (stored repeatedly in the first 6 entries)
        joint_code = np.clip(
            np.round((node_array[:6].mean() + 0.5) * 5.0), 1, 5
        ).astype(int)
        joint_type = joint_ref["bwd"].get(joint_code, "fixed")

        # semantic label (last 6 entries)
        label_code = np.clip(
            np.round((node_array[18:].mean() + 0.8) * 5.0), 0, 7
        ).astype(int)
        part_name = sem_ref["bwd"].get(label_code, "unknown")

        # joint axis direction and origin
        axis_dir = node_array[6:9] / 0.7
        axis_ori = node_array[9:12]

        # joint range repeats every value three times -> take the mean of each block
        raw_range = np.stack(
            [
                node_array[12:15].mean(),
                node_array[15:18].mean(),
            ]
        )

        if joint_type == "fixed":
            joint_range = [0.0, 0.0]
            axis_dir = np.zeros(3, dtype=np.float32)
        elif joint_type in ("revolute", "continuous"):
            angle = float(raw_range[0] * 360.0)
            if joint_type == "continuous":
                joint_range = [0.0, 360.0]
            else:
                joint_range = [0.0, angle]
        elif joint_type in ("prismatic", "screw"):
            extent = float(raw_range[1])
            joint_range = [0.0, extent]
        else:
            joint_range = [0.0, 0.0]

        node_info = {
            "name": part_name,
            "joint": {
                "type": joint_type,
                "axis": {
                    "direction": axis_dir.tolist(),
                    "origin": axis_ori.tolist(),
                },
                "range": joint_range,
            },
        }

        if mask is not None and joint_type == "fixed" and mask[6:18].sum() > 0:
            # Preserve masked-out values if they carry information for downstream use.
            node_info["joint"]["axis"]["direction"] = (node_array[6:9] / 0.7).tolist()
            node_info["joint"]["axis"]["origin"] = node_array[9:12].tolist()

        return node_info

    def get_instance(self, instance_path):
        if instance_path in self.data_cache_slat_part:
            return self.data_cache_slat_part[instance_path]
        try:

            part_info_sorted, bbox = prepaer_parts_meta(
                instance_path,
                part_info_path=self.part_info_file,
                transform_info_path=self.transform_info_file,
            )
            ###
            slat_dict = self.make_omnipart_training_slat(
                instance_path, part_info_sorted
            )
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
            print(f"Error processing {os.path.basename(instance_path)}:\n {e}")
            # raise e
            self.metadata.drop(index=instance_path, inplace=True)
            return None
        slat = slat_dict["slat"]
        part_layouts = slat_dict["part_layouts"]
        # data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))

        # coords = torch.tensor(data['coords']).int()
        # feats = torch.tensor(data['feats']).float()
        # SparseTensor()
        if self.normalization is not None:  ### NOTE:: last one from omnipart
            slat.feats[:, :8] = (slat.feats[:, :8] - self.mean) / self.std
        self.data_cache_slat_part[instance_path] = {
            "coords": slat.coords,
            "feats": slat.feats,
            "part_layouts": part_layouts,
            "part_info": part_info_sorted,
            "x_0_arti": arti_info if self.is_predict_arti_info else None,
            "x_0_arti_mask": arti_info_mask if self.is_predict_arti_info else None,
        }

        return self.data_cache_slat_part[instance_path]

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]  #  [[0, 1]， [2, 3]]
        else:
            group_idx = load_balanced_group_indices(
                [b["coords"].shape[0] for b in batch], split_size
            )
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            coords_art = []
            feats_art = []
            layout_art = []
            part_layouts = []
            start = 0
            start_art = 0
            for i, b in enumerate(sub_batch):
                coords.append(
                    torch.cat(
                        [
                            torch.full((b["coords"].shape[0], 1), i, dtype=torch.int32),
                            b["coords"],
                        ],
                        dim=-1,
                    )
                )
                feats.append(b["feats"])
                layout.append(slice(start, start + b["coords"].shape[0]))
                start += b["coords"].shape[0]
                ######
                coords_art.append(
                    torch.full((b["x_0_arti"].shape[0], 1), i, dtype=torch.int32)
                )
                feats_art.append(torch.cat([b["x_0_arti"], b["x_0_arti_mask"]], dim=-1))
                layout_art.append(slice(start_art, start_art + b["x_0_arti"].shape[0]))
                start_art += b["x_0_arti"].shape[0]
                #######

                layout_part = b.get("part_layouts", None)
                assert (
                    layout_part is not None
                ), "part_layouts should be provided in omnipart mode!"
                part_layouts.append(layout_part)  ###  NOTE:

            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack["x_0"] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack["x_0"]._shape = torch.Size(
                [len(group), *sub_batch[0]["feats"].shape[1:]]
            )
            pack["x_0"].register_spatial_cache("layout", layout)
            pack["part_layouts_"] = part_layouts

            pack["x_0_arti"] = SparseTensor(
                coords=torch.cat(coords_art),
                feats=torch.cat(feats_art),
            )
            assert (
                pack["x_0_arti"].shape[0] == pack["x_0"].shape[0]
            ), f"pack['x_0'] and pack['x_0_arti'] should have the same batch size! { pack['x_0'].shape[0]}, { pack['x_0_arti'].shape[0]}"
            assert (
                pack["x_0_arti"].layout == layout_art
            ), f"pack['x_0_arti'] and layout_art should be consistent! { pack['x_0_arti'].layout}, {layout_art}"
            # collate other data
            keys = [
                k
                for k in sub_batch[0].keys()
                if k
                not in [
                    "coords",
                    "feats",
                    "x_0_arti",
                    "x_0_arti_mask",
                ]
            ]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = [b[k] for b in sub_batch]
                else:
                    pack[k] = [b[k] for b in sub_batch]
            assert (
                pack["part_layouts_"] == pack["part_layouts"]
            ), "part_layouts should be consistent!"
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


from modules.inference_utils import change_pcd_range


class TextConditionedSLat(TextConditionedMixin, SLat):
    """
    Text conditioned structured latent dataset
    """

    pass


class ImageConditionedSLat(ImageConditionedMixin, SLat):
    """
    Image conditioned structured latent dataset
    """

    pass


# class ImageConditioned_ArtiSLat(ImageConditionedMixin_PartNet, Arti_SLat):
#     """
#     Image conditioned structured latent dataset
#     """

#     pass
