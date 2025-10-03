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
import math
import numpy as np


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


### 计算 ∫ SH(ω) * SG(ω; μ, λ) dω 的解析解 实现开始 ###
### 支持可变SH阶数(0-4阶) ###

def evaluate_sh_basis(dirs, deg=2):
    """
    评估球谐基函数在给定方向上的值。
    Args:
        dirs: (..., 3) - 归一化的方向向量 (x, y, z)
        deg: int - SH阶数 (0-4), 默认为2
    Returns:
        (..., (deg+1)^2) - SH基函数的值
    """
    assert deg >= 0 and deg <= 4, "Only degrees 0-4 are supported"

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    # 计算需要的系数数量
    num_coeffs = (deg + 1) ** 2
    output_shape = dirs.shape[:-1]
    sh = torch.zeros((*output_shape, num_coeffs), device=dirs.device, dtype=dirs.dtype)

    # L=0
    sh[..., 0] = C0  # 0.28209479177387814

    if deg >= 1:
        # L=1
        sh[..., 1] = -C1 * y  # Y_1^-1
        sh[..., 2] = C1 * z  # Y_1^0
        sh[..., 3] = -C1 * x  # Y_1^1

    if deg >= 2:
        # L=2
        sh[..., 4] = C2[0] * x * y  # Y_2^-2
        sh[..., 5] = C2[1] * y * z  # Y_2^-1
        sh[..., 6] = C2[2] * (3.0 * z * z - 1.0)  # Y_2^0
        sh[..., 7] = C2[3] * x * z  # Y_2^1
        sh[..., 8] = C2[4] * (x * x - y * y)  # Y_2^2

    if deg >= 3:
        # L=3
        sh[..., 9] = C3[0] * y * (3 * x * x - y * y)
        sh[..., 10] = C3[1] * x * y * z
        sh[..., 11] = C3[2] * y * (4 * z * z - x * x - y * y)
        sh[..., 12] = C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y)
        sh[..., 13] = C3[4] * x * (4 * z * z - x * x - y * y)
        sh[..., 14] = C3[5] * z * (x * x - y * y)
        sh[..., 15] = C3[6] * x * (x * x - 3 * y * y)

    if deg >= 4:
        # L=4
        sh[..., 16] = C4[0] * x * y * (x * x - y * y)
        sh[..., 17] = C4[1] * y * z * (3 * x * x - y * y)
        sh[..., 18] = C4[2] * x * y * (7 * z * z - 1)
        sh[..., 19] = C4[3] * y * z * (7 * z * z - 3)
        sh[..., 20] = C4[4] * (z * z * (35 * z * z - 30) + 3)
        sh[..., 21] = C4[5] * x * z * (7 * z * z - 3)
        sh[..., 22] = C4[6] * (x * x - y * y) * (7 * z * z - 1)
        sh[..., 23] = C4[7] * x * z * (x * x - 3 * y * y)
        sh[..., 24] = C4[8] * (
            x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y)
        )

    return sh


def compute_sg_zh_analytical(lambda_val, deg=2):
    """
    使用解析公式计算SG的Zonal Harmonics (ZH)系数

    基于Legendre多项式与exp(λx)积分的推导

    Args:
        lambda_val: (...) - SG锐度参数λ
        deg: int - SH阶数 (0-4), 默认为2
    Returns:
        list of tensors - [zh0, zh1, ..., zh_deg]
    """
    assert deg >= 0 and deg <= 4, "Only degrees 0-4 are supported"

    device = lambda_val.device
    dtype = lambda_val.dtype

    # 防止除零
    lam = lambda_val + 1e-8

    # 预计算常用项
    lam_inv = 1.0 / lam
    lam_inv_sq = lam_inv * lam_inv
    lam_inv_cub = lam_inv_sq * lam_inv
    lam_inv_4 = lam_inv_cub * lam_inv
    lam_inv_5 = lam_inv_4 * lam_inv

    exp_minus_lam = torch.exp(-lam)
    sinh_lam = torch.sinh(lam)
    cosh_lam = torch.cosh(lam)

    zh_coeffs = []

    # L=0: P_0(x) = 1
    # I_0 = 2sinh(λ)/λ
    factor0 = 2.0 * math.sqrt(math.pi)
    zh0 = factor0 * exp_minus_lam * sinh_lam * lam_inv
    zh_coeffs.append(zh0)

    if deg >= 1:
        # L=1: P_1(x) = x
        # I_1 = 2cosh(λ)/λ - 2sinh(λ)/λ²
        # 除以2后: cosh(λ)/λ - sinh(λ)/λ²
        factor1 = 2.0 * math.sqrt(3.0 * math.pi)
        zh1 = factor1 * exp_minus_lam * (cosh_lam * lam_inv - sinh_lam * lam_inv_sq)
        zh_coeffs.append(zh1)

    if deg >= 2:
        # L=2: P_2(x) = (3x² - 1)/2
        # 积分/2 = (1/λ + 3/λ³)sinh(λ) - 3/λ²cosh(λ)
        factor2 = 2.0 * math.sqrt(5.0 * math.pi)
        term_sinh = (lam_inv + 3.0 * lam_inv_cub) * sinh_lam
        term_cosh = 3.0 * lam_inv_sq * cosh_lam
        zh2 = factor2 * exp_minus_lam * (term_sinh - term_cosh)
        zh_coeffs.append(zh2)

    if deg >= 3:
        # L=3: P_3(x) = (5x³ - 3x)/2
        # 积分/2 = (1/λ + 15/λ³)cosh(λ) - (6/λ² + 15/λ⁴)sinh(λ)
        factor3 = 2.0 * math.sqrt(7.0 * math.pi)
        term_cosh = (lam_inv + 15.0 * lam_inv_cub) * cosh_lam
        term_sinh = (6.0 * lam_inv_sq + 15.0 * lam_inv_4) * sinh_lam
        zh3 = factor3 * exp_minus_lam * (term_cosh - term_sinh)
        zh_coeffs.append(zh3)

    if deg >= 4:
        # L=4: P_4(x) = (35x⁴ - 30x² + 3)/8
        # 积分/2 = (1/λ + 45/λ³ + 105/λ⁵)sinh(λ) - (10/λ² + 105/λ⁴)cosh(λ)
        factor4 = 2.0 * math.sqrt(9.0 * math.pi)
        term_sinh = (lam_inv + 45.0 * lam_inv_cub + 105.0 * lam_inv_5) * sinh_lam
        term_cosh = (10.0 * lam_inv_sq + 105.0 * lam_inv_4) * cosh_lam
        zh4 = factor4 * exp_minus_lam * (term_sinh - term_cosh)
        zh_coeffs.append(zh4)

    return zh_coeffs


def sh_sg_integral(sh_coeffs, sg_axis, sg_sharpness, deg=None):
    """
    计算 ∫ SH(ω) * SG(ω; μ, λ) dω 的解析解

    核心思路：
    1. 将SG展开为zonal harmonics (ZH)
    2. 利用SH旋转将ZH旋转到μ方向
    3. 使用SH正交性完成积分

    基于文献：
    - "Stupid Spherical Harmonics Tricks" - Peter-Pike Sloan
    - Wang et al. "All-Frequency Rendering of Dynamic, Spatially-Varying Reflectance"

    Args:
        sh_coeffs: (H, W, 3, N) - SH系数，N=(deg+1)^2，pytorch tensor
        sg_axis: (H, W, 3) - SG中心方向（已归一化），pytorch tensor
        sg_sharpness: (H, W, 1) - SG锐度参数λ，pytorch tensor
        deg: int or None - SH阶数 (0-4)。如果为None，则从sh_coeffs推断

    Returns:
        (H, W, 3) - 积分结果，pytorch tensor
    """
    H, W, C, num_coeffs = sh_coeffs.shape
    device = sh_coeffs.device
    dtype = sh_coeffs.dtype

    # 推断SH阶数
    if deg is None:
        deg = int(math.sqrt(num_coeffs)) - 1
        assert (
            deg + 1
        ) ** 2 == num_coeffs, f"Invalid number of SH coefficients: {num_coeffs}"

    assert deg >= 0 and deg <= 4, "Only degrees 0-4 are supported"
    assert (
        num_coeffs == (deg + 1) ** 2
    ), f"sh_coeffs shape mismatch: expected {(deg+1)**2} coeffs for deg={deg}, got {num_coeffs}"

    # 确保方向是归一化的
    sg_axis = sg_axis / (torch.norm(sg_axis, dim=-1, keepdim=True) + 1e-8)
    lam = sg_sharpness.squeeze(-1)  # (H, W)

    # 计算SG的ZH系数，使用解析解
    zh_coeffs = compute_sg_zh_analytical(lam, deg=deg)

    # 在μ方向评估所有SH基函数
    mu_basis = evaluate_sh_basis(sg_axis, deg=deg)  # (H, W, (deg+1)^2)

    # 构造旋转后的SG的SH系数
    sg_sh = torch.zeros(H, W, num_coeffs, device=device, dtype=dtype)

    # 对每个阶数l，应用旋转公式
    # c'_lm = sqrt(4π/(2l+1)) * Y_l^m(μ) * zh_l
    idx = 0
    for l in range(deg + 1):
        factor_l = math.sqrt(4.0 * math.pi / (2.0 * l + 1.0))
        num_m = 2 * l + 1  # 每个阶数l有2l+1个m值

        # 对于这个l阶的所有m: -l, ..., 0, ..., l
        for m in range(num_m):
            sg_sh[..., idx] = zh_coeffs[l] * factor_l * mu_basis[..., idx]
            idx += 1

    # 积分: 利用SH的正交性，两个函数在球面上的积分等于其SH系数的点积
    # ∫ f(ω)g(ω)dω = Σ_{l,m} f_lm * g_lm
    result = torch.einsum("hwci,hwi->hwc", sh_coeffs, sg_sh)

    return result

### 计算 ∫ SH(ω) * SG(ω; μ, λ) dω 的解析解 实现结束 ###


def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
