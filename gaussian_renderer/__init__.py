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
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.sh_utils import sh_to_irradiance 


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # --- FIX START ---
    # Ensure scales and rotations are always provided, and cov3D_precomp is None
    # (unless specifically using that path, which we are not).
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None
    # The original if/else for pipe.compute_cov3D_python is complex and not needed
    # for our current goal. We simplify it to always use scales/rotations.
    # --- FIX END ---
    
    # --- MODIFICATION START ---
    # Our new strategy:
    # 1. We will NOT precompute colors. We will blend material and light properties.
    # 2. We pack our custom attributes (albedo, light_sh) into the SH feature buffer.
    #    The rasterizer's 'shs' input can carry arbitrary per-Gaussian data.
    
    pipe.convert_SHs_python = False # Ensure this is false
    shs = None
    colors_precomp = None

    if override_color is None:
        # Pack Albedo (3 channels) and Light SH (up to 16*3 channels)
        # We'll use a trick: store Albedo in a separate buffer and blend it as 'color'
        # and Light SH in the 'shs' buffer.
        # This is slightly inefficient as we blend two things, but simpler to implement.
        
        # Alternative & better: Pack everything into one buffer. Let's do that.
        # The 'shs' buffer has shape (N, (max_sh_degree+1)**2, 3). We have 16 bands for degree 3.
        # Band 0: Diffuse Color (Albedo)
        # Bands 1-15: Light SH coeffs
        
        num_points = pc.get_xyz.shape[0]
        max_sh_bands = (pc.max_sh_degree + 1) ** 2
        
        # We need to ensure pc.get_light_sh() returns enough bands, even if active_sh_degree is low.
        # Let's get the full SH tensor (N, 16, 3)
        light_sh_full = torch.cat((pc._features_dc, pc._features_rest), dim=1)

        # Create a packed tensor for blending
        packed_features = torch.zeros(num_points, max_sh_bands + 1, 3, device="cuda")
        packed_features[:, 0, :] = pc.get_diffuse_color
        packed_features[:, 1:, :] = light_sh_full

        # The rasterizer expects shape (N, C, n_feats), we give it (N, 3, 17)
        shs = packed_features.transpose(1, 2).contiguous()

    else:
        # Keep override_color functionality for debugging/visualization
        colors_precomp = override_color

    # Call the rasterizer. The first return value will be our blended packed_features.
    blended_packed_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # --- Python-side shading logic ---
    if override_color is None:
        # Unpack the blended attributes from the rendered image
        # Shape: (C, H, W) where C = 3 * (17)
        num_bands = max_sh_bands + 1
        blended_packed_features = blended_packed_features.view(3, num_bands, blended_packed_features.shape[1], blended_packed_features.shape[2])

        blended_albedo = blended_packed_features[:, 0, :, :]                             # Shape (3, H, W)
        blended_light_sh = blended_packed_features[:, 1:max_sh_bands+1, :, :]            # Shape (3, 16, H, W)

        # Get the blended world-space normals rendered by the rasterizer
        render_normal_view = allmap[2:5] # These are in view space
        render_normal_world = (render_normal_view.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

        # Active SH degree determines how many SH bands to use for lighting
        active_sh_bands = (pc.active_sh_degree + 1)**2
        
        # Calculate irradiance. sh_to_irradiance expects (..., C, num_bands) and (..., 3)
        # We need to select the active bands and reshape
        irradiance = sh_to_irradiance(
            blended_light_sh[:, :active_sh_bands, :, :].permute(2, 3, 0, 1), # -> (H, W, 3, active_bands)
            render_normal_world.permute(1, 2, 0)                             # -> (H, W, 3)
        )
        irradiance = irradiance.permute(2, 0, 1) # -> (3, H, W)
        
        # Lambertian shading: color = albedo/pi * irradiance
        # The rasterizer output is alpha-premultiplied, so we compute the RGB part
        # and the final color will be RGB * alpha.
        shaded_rgb = blended_albedo / torch.pi * irradiance
        shaded_rgb = torch.clamp(shaded_rgb, min=0.0) # Ensure no negative light
        
        # Combine with background
        render_alpha = allmap[1:2]
        rendered_image = shaded_rgb + (1.0 - render_alpha) * bg_color.unsqueeze(-1).unsqueeze(-1)
    else:
        rendered_image = blended_packed_features # This is the case where override_color was used
    # --- MODIFICATION END ---
    
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets