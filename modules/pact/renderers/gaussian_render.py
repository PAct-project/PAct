#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float,
    far: float,
    ) -> torch.Tensor:
    """
    Convert OpenCV-style camera intrinsics matrix to OpenGL perspective projection matrix.
    
    This function transforms a standard 3x3 camera intrinsics matrix into a 4x4 perspective
    projection matrix compatible with OpenGL rendering pipeline. The resulting matrix
    properly handles the coordinate system differences between computer vision and
    computer graphics conventions.
    
    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix containing focal lengths
                    and principal point coordinates
        near (float): Distance to the near clipping plane (must be positive)
        far (float): Distance to the far clipping plane (must be greater than near)
    
    Returns:
        torch.Tensor: [4, 4] OpenGL perspective projection matrix for rendering
    """
    
    # Extract focal lengths and principal point from intrinsics matrix
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Focal lengths in x and y directions
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # Principal point coordinates
    
    # Initialize empty 4x4 projection matrix
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    
    # Fill in the projection matrix components
    ret[0, 0] = 2 * fx  # Scale for x axis based on horizontal focal length
    ret[1, 1] = 2 * fy  # Scale for y axis based on vertical focal length
    ret[0, 2] = 2 * cx - 1  # X offset based on principal point (OpenCV to OpenGL conversion)
    ret[1, 2] = - 2 * cy + 1  # Y offset based on principal point (with flipped Y axis)
    ret[2, 2] = far / (far - near)  # Handle depth mapping to clip space
    ret[2, 3] = near * far / (near - far)  # Term for perspective division in clip space
    ret[3, 2] = 1.  # Enable perspective division
    
    return ret

def render(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene using 3D Gaussians.
    
    This function performs the rasterization of 3D Gaussian points into a 2D image from a given viewpoint.
    
    Args:
        viewpoint_camera: Camera parameters including position, view transform, and projection
        pc (Gaussian): Point cloud represented as 3D Gaussians
        pipe: Pipeline configuration parameters
        bg_color (torch.Tensor): Background color tensor (must be on GPU)
        scaling_modifier (float): Scale modifier for the Gaussian splats
        override_color (torch.Tensor, optional): Custom colors to override computed SH-based colors
    
    Returns:
        edict: Dictionary containing rendered image, viewspace points, visibility filter, and radii information
    """
    # Lazy import of the rasterization module to avoid circular dependencies
    # or to improve startup performance when not needed immediately
    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # Create zero tensor for screen space points
    # This tensor will hold gradients of the 2D (screen-space) means for optimization
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
        
    # Calculate camera frustum parameters from the field of view
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # Get kernel size from the pipeline configuration
    kernel_size = pipe.kernel_size
    
    # Initialize subpixel offset for all pixels (used for anti-aliasing)
    subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), 
                                dtype=torch.float32, device="cuda")

    # Configure the Gaussian rasterization settings with all necessary parameters
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    # Create the rasterizer with the configured settings
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get the Gaussian 3D positions and opacities
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Handle covariance computation options
    # Either use precomputed 3D covariance or let the rasterizer compute it from scales and rotations
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # Compute 3D covariances in Python before rasterization
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # Let the rasterizer compute covariances from scale and rotation
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Handle color computation options
    # Either use override colors, precomputed colors from SHs, or let the rasterizer compute colors from SHs
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # Convert spherical harmonics to RGB colors in Python
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # Calculate the view direction from Gaussian center to camera
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # Evaluate spherical harmonics to get RGB colors
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # Apply offset and clamp to ensure valid color values
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # Let the rasterizer convert SHs to colors
            shs = pc.get_features
    else:
        # Use provided override colors
        colors_precomp = override_color

    # Perform the rasterization to generate the final rendered image
    # This projects the 3D Gaussians to 2D and blends them according to their opacities
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    # Return the rendering results in a dictionary
    # radii > 0 creates a filter for visible Gaussians (those not frustum-culled)
    return edict({"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii})

class GaussianRenderer:
    """
    A renderer for Gaussian Splatting that converts 3D Gaussian primitives into 2D images.
    
    This renderer projects 3D Gaussian splats onto a 2D image plane using the provided
    camera parameters, handling the rasterization process through an optimized backend.
    
    Args:
        rendering_options (dict): Configuration options for rendering including resolution,
                                    depth range, background color, and supersampling level.
    """

    def __init__(self, rendering_options={}) -> None:
        # Initialize default pipeline parameters
        self.pipe = edict({
            "kernel_size": 0.1,       # Size of the Gaussian kernel for rasterization
            "convert_SHs_python": False,  # Whether to convert Spherical Harmonics to colors in Python
            "compute_cov3D_python": False,  # Whether to compute 3D covariance matrices in Python
            "scale_modifier": 1.0,    # Global scaling factor for all Gaussians
            "debug": False            # Enable/disable debug mode
        })
        
        # Initialize default rendering options
        self.rendering_options = edict({
            "resolution": None,       # Output image resolution (width and height)
            "near": None,             # Near clipping plane distance
            "far": None,              # Far clipping plane distance
            "ssaa": 1,                # Super-sampling anti-aliasing factor (1 = disabled)
            "bg_color": 'random',     # Background color ('random' or specific color)
        })
        
        # Update with user-provided options
        self.rendering_options.update(rendering_options)
        
        # Initialize background color (will be set during rendering)
        self.bg_color = None
    
    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the 3D Gaussian representation from a given camera viewpoint.

        This method projects the 3D Gaussians onto a 2D image plane using the provided camera parameters,
        handling the full rendering pipeline including projection, rasterization, and optional supersampling.

        Args:
            gaussian: The Gaussian representation containing positions, features, and other attributes
            extrinsics (torch.Tensor): (4, 4) camera extrinsics matrix defining camera position and orientation
            intrinsics (torch.Tensor): (3, 3) camera intrinsics matrix with focal lengths and principal point
            colors_overwrite (torch.Tensor): Optional (N, 3) tensor to override Gaussian colors

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        # Extract rendering parameters from options
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]  # Super-sampling anti-aliasing factor
        
        # Set background color based on rendering options
        if self.rendering_options["bg_color"] == 'random':
            # Randomly choose either black or white background
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            # Use specified background color
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        # Prepare camera parameters for the renderer
        view = extrinsics  # World-to-camera transform
        
        # Convert OpenCV intrinsics to OpenGL projection matrix
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        # Extract camera center from extrinsics (inverse of view matrix)
        camera = torch.inverse(view)[:3, 3]
        
        # Calculate field of view from focal lengths
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)  # Horizontal FoV in radians
        fovy = 2 * torch.atan(0.5 / focaly)  # Vertical FoV in radians
            
        # Build complete camera parameter dictionary
        camera_dict = edict({
            "image_height": resolution * ssaa,  # Apply supersampling if enabled
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),  # Transpose for OpenGL convention
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),  # Combined projection and view
            "camera_center": camera
        })

        # Perform the actual rendering using the 3D Gaussian rasterizer
        render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, 
                            override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)

        # Handle supersampling by downsampling the high-resolution render to the target resolution
        if ssaa > 1:
            # Use bilinear interpolation with antialiasing to downsample the image
            render_ret.render = F.interpolate(render_ret.render[None], 
                                            size=(resolution, resolution), 
                                            mode='bilinear', 
                                            align_corners=False, 
                                            antialias=True).squeeze()
            
        # Return the final rendered color image
        ret = edict({
            'color': render_ret['render']
        })
        return ret
