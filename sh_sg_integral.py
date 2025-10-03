import torch
import math


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
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


### 计算 ∫ SH(ω) * SG(ω; μ, λ) dω 的解析解 实现开始 ###
### 已完成：支持可变SH阶数(0-4阶) ###


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
    使用解析公式计算SG的Zonal Harmonics (ZH)系数。
    该SG的定义为 exp(λ(cos(θ) - 1))。

    Args:
        lambda_val: (H, W) 或 (...) - SG锐度参数λ
        deg: int - SH阶数 (0-4), 默认为2

    Returns:
        list of tensors - [zh0, zh1, ..., zh_deg], 每个shape与lambda_val相同
    """
    assert deg >= 0 and deg <= 4, "Only degrees 0-4 are supported"

    device = lambda_val.device
    dtype = lambda_val.dtype

    # 为防止除以零，添加一个小的epsilon
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

    # L=0
    # zh_0 = 2 * sqrt(π) * e^{-λ} * sinh(λ) / λ
    factor0 = 2.0 * math.sqrt(math.pi)
    zh0 = factor0 * exp_minus_lam * sinh_lam * lam_inv
    zh_coeffs.append(zh0)

    if deg >= 1:
        # L=1
        # zh_1 = 2 * sqrt(3π) * e^{-λ} * [cosh(λ)/λ - sinh(λ)/λ²]
        factor1 = 2.0 * math.sqrt(3.0 * math.pi)
        zh1 = factor1 * exp_minus_lam * (cosh_lam * lam_inv - sinh_lam * lam_inv_sq)
        zh_coeffs.append(zh1)

    if deg >= 2:
        # L=2
        # zh_2 = 2 * sqrt(5π) * e^{-λ} * [(1/λ + 3/λ³)sinh(λ) - (3/λ²)cosh(λ)]
        factor2 = 2.0 * math.sqrt(5.0 * math.pi)
        term_sinh = (lam_inv + 3.0 * lam_inv_cub) * sinh_lam
        term_cosh = 3.0 * lam_inv_sq * cosh_lam
        zh2 = factor2 * exp_minus_lam * (term_sinh - term_cosh)
        zh_coeffs.append(zh2)

    if deg >= 3:
        # L=3
        # zh_3 = 2 * sqrt(7π) * e^{-λ} * [(6/λ² + 15/λ⁴)cosh(λ) - (6/λ + 15/λ³)sinh(λ)]
        factor3 = 2.0 * math.sqrt(7.0 * math.pi)
        term_cosh = (6.0 * lam_inv_sq + 15.0 * lam_inv_4) * cosh_lam
        term_sinh = (6.0 * lam_inv + 15.0 * lam_inv_cub) * sinh_lam
        zh3 = factor3 * exp_minus_lam * (term_cosh - term_sinh)
        zh_coeffs.append(zh3)

    if deg >= 4:
        # L=4
        # zh_4 = 2 * sqrt(9π) * e^{-λ} * [(1/λ + 45/λ³ + 105/λ⁵)sinh(λ) - (10/λ² + 105/λ⁴)cosh(λ)]
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
