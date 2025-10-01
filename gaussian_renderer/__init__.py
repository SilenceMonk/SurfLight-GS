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
from utils.sh_utils import eval_sh, sh_to_irradiance
from utils.point_utils import depth_to_normal


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene with material-light decomposition.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
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
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = (
            torch.tensor(
                [
                    [W / 2, 0, 0, (W - 1) / 2],
                    [0, H / 2, 0, (H - 1) / 2],
                    [0, 0, far - near, near],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .cuda()
            .T
        )
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]])
            .permute(0, 2, 1)
            .reshape(-1, 9)
        )
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # --- CRITICAL MODIFICATION: Material-Light Decomposition ---

    # Get the SH coefficients for incident lighting
    shs = None
    colors_precomp = None

    if override_color is None:
        # Get diffuse albedo (base color)
        diffuse_color = pc.get_diffuse_color  # Shape: (N, 3), range [0,1]

        # Get incident light SH coefficients
        light_sh = pc.get_light_sh  # Shape: (N, num_sh_bands, 3)

        if pipe.convert_SHs_python:
            # Compute view-dependent lighting in Python
            # This path evaluates SH in camera space

            # Get the normals from the 2D Gaussian disks
            # Normal is t_u × t_v, derived from rotation
            rotations_full = pc.get_rotation  # (N, 4) quaternion

            # Extract normal from rotation (third column of rotation matrix)
            # For 2DGS, the normal is the cross product of the two tangent vectors
            # This is implicitly the third basis vector from the rotation
            # We can get it from the rotation quaternion
            from utils.general_utils import build_rotation

            R = build_rotation(pc._rotation)  # (N, 3, 3)
            normals = R[:, :, 2]  # (N, 3) - third column is the normal
            normals = torch.nn.functional.normalize(normals, dim=-1)

            # Transform normals to world space (they're already in world space for 2DGS)
            # No transformation needed as 2DGS normals are in world coordinates

            # Compute irradiance from SH lighting
            # light_sh shape: (N, num_sh_bands, 3)
            # We need to transpose to (N, 3, num_sh_bands) for sh_to_irradiance
            light_sh_transposed = light_sh.transpose(1, 2)  # (N, 3, num_sh_bands)

            # Compute irradiance: ∫ L_in(ω) * cos(θ) dω
            irradiance = sh_to_irradiance(light_sh_transposed, normals)  # (N, 3)

            # Apply Lambertian BRDF: f_r = albedo / π
            # Final color = albedo * irradiance / π
            colors_precomp = diffuse_color * irradiance / math.pi
            colors_precomp = torch.clamp(colors_precomp, min=0.0, max=1.0)

        else:
            # Let the CUDA rasterizer handle SH evaluation
            # Pass the light SH coefficients to the rasterizer
            # The rasterizer will evaluate them per-pixel

            # IMPORTANT: The original rasterizer expects SH for view-dependent color
            # We need to modify the rasterizer to support our light-material model
            # For now, we'll do the evaluation in Python for correctness

            # Fallback to Python evaluation
            rotations_full = pc.get_rotation
            from utils.general_utils import build_rotation

            R = build_rotation(pc._rotation)
            normals = R[:, :, 2]
            normals = torch.nn.functional.normalize(normals, dim=-1)

            light_sh_transposed = light_sh.transpose(1, 2)
            irradiance = sh_to_irradiance(light_sh_transposed, normals)
            colors_precomp = diffuse_color * irradiance / math.pi
            colors_precomp = torch.clamp(colors_precomp, min=0.0, max=1.0)
    else:
        colors_precomp = override_color

    # --- END MODIFICATION ---

    # Render
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,  # None when using precomputed colors
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    # Additional regularizations
    render_alpha = allmap[1:2]

    # Get normal map
    # Transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0)
        @ (viewpoint_camera.world_view_transform[:3, :3].T)
    ).permute(2, 0, 1)

    # Get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # Get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # Get depth distortion map
    render_dist = allmap[6:7]

    # Pseudo surface attributes
    surf_depth = (
        render_depth_expected * (1 - pipe.depth_ratio)
        + (pipe.depth_ratio) * render_depth_median
    )

    # Generate pseudo surface normal for regularizations
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        }
    )

    return rets
