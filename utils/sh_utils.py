#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F  # Make sure F is imported

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def sh_to_irradiance(sh, normals):
    """
    Calculates irradiance from up to 2nd-order SH light and a normal map.
    Handles variable SH degrees based on input tensor shape.

    Args:
        sh (torch.Tensor): SH coefficients of incident light. Shape (..., C, num_bands),
                           e.g., (H, W, 3, 1), (H, W, 3, 4), or (H, W, 3, 9).
        normals (torch.Tensor): Normal vectors. Shape (..., 3)

    Returns:
        torch.Tensor: Irradiance (RGB). Shape (..., 3)
    """
    # ... (Constants C0, C1, etc. remain the same)
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = 1.0925484305920792
    C3 = 0.31539156525252005
    C4 = 0.5462742152960396

    num_bands = sh.shape[-1]

    # Ensure normals are unit vectors and have the right shape for broadcasting
    normals = F.normalize(normals, dim=-1)
    nx, ny, nz = normals[..., 0:1], normals[..., 1:2], normals[..., 2:3]

    # Initialize irradiance with zeros
    irradiance = torch.zeros_like(normals)

    # Always compute L0 (ambient) term
    if num_bands >= 1:
        # NOTE: The formula for irradiance involves pi. The original 2DGS code might not use it.
        # Let's use the simplified form that's common in games, which absorbs constants.
        # Irradiance_L0 = c1 * L0,0 where c1 = sqrt(pi).
        # We need to be consistent. Let's use the formula from "Stupid SH Tricks".
        # Irradiance(n) = sum_l,m A_l * L_lm * Y_lm(n)
        # For Lambertian: A0=pi, A1=2pi/3, A2=pi/4.

        # Let's simplify and assume constants are baked into the learned SH coefficients for now.
        # This is often more stable for learning.
        # This means we just need the dot product with the SH basis functions.
        # The basis functions Y_lm(n) are needed.
        # Y_0,0 = C0
        # Y_1,-1 = -C1 * y, Y_1,0 = C1 * z, Y_1,1 = -C1 * x
        # ... and so on.

        # Let's implement this correctly.
        irradiance += sh[..., :, 0] * C0

    if num_bands >= 4:  # L1 bands (1-3)
        irradiance += sh[..., :, 1] * (-C1 * ny)
        irradiance += sh[..., :, 2] * (C1 * nz)
        irradiance += sh[..., :, 3] * (-C1 * nx)

    if num_bands >= 9:  # L2 bands (4-8)
        irradiance += sh[..., :, 4] * (C2 * nx * ny)
        irradiance += sh[..., :, 5] * (C2 * -ny * nz)
        irradiance += sh[..., :, 6] * (C3 * (3 * nz * nz - 1))
        irradiance += sh[..., :, 7] * (C2 * -nx * nz)
        irradiance += sh[..., :, 8] * (C4 * (nx * nx - ny * ny))

    # The factor of pi is part of the Lambertian BRDF (albedo / pi).
    # The integral ∫ L_in(ω) * cos(θ) dω is the irradiance.
    # The result here should be multiplied by albedo / pi.
    # To avoid negative irradiance, clamp it.
    return torch.clamp(irradiance, min=0.0)


def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
