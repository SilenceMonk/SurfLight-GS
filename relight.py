import torch
import os
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
import numpy as np
from PIL import Image
from tqdm import tqdm
import math


def compute_lambertian_sh_kernel(normals):
    """
    Lambertian kernel的SH投影
    已经包含了 max(0, n·ω) 的卷积结果
    """
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

    sh_kernel = torch.zeros((normals.shape[0], 9), device=normals.device)

    # 这些系数来自 ∫ Y_lm(ω) * max(0, n·ω) dω 的解析解
    # 参考: Ramamoorthi & Hanrahan 2001

    # L=0
    sh_kernel[:, 0] = 0.886227  # π * Y_0,0 = π * 0.282095

    # L=1
    sh_kernel[:, 1] = 1.023328 * ny  # (2π/3) * Y_1,-1
    sh_kernel[:, 2] = 1.023328 * nz  # (2π/3) * Y_1,0
    sh_kernel[:, 3] = 1.023328 * nx  # (2π/3) * Y_1,1

    # L=2
    sh_kernel[:, 4] = 0.858086 * nx * ny
    sh_kernel[:, 5] = 0.858086 * ny * nz
    sh_kernel[:, 6] = 0.247708 * (3 * nz * nz - 1)
    sh_kernel[:, 7] = 0.858086 * nx * nz
    sh_kernel[:, 8] = 0.429043 * (nx * nx - ny * ny)

    return sh_kernel


def render_with_global_lighting(viewpoint_camera, pc, pipe, bg_color, env_sh):
    """
    使用全局环境光照渲染

    Args:
        env_sh: (3, 9) 全局环境光的SH系数,每个RGB通道9个系数(2阶SH)
    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from utils.general_utils import build_rotation

    # 基础光栅化设置
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
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
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # === 核心修改:使用全局光照 ===

    # 1. 获取diffuse albedo
    diffuse_color = pc.get_diffuse_color  # (N, 3)

    # 2. 获取每个Gaussian的法线(从rotation提取)
    R = build_rotation(pc._rotation)  # (N, 3, 3)
    normals = R[:, :, 2]  # (N, 3) - 第三列是法线
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # 在render_with_global_lighting函数中,替换kernel计算部分:
    sh_kernel = compute_lambertian_sh_kernel(normals)  # 使用上面的函数

    # irradiance计算保持不变
    irradiance = torch.zeros((normals.shape[0], 3), device=normals.device)
    for c in range(3):
        irradiance[:, c] = torch.sum(sh_kernel * env_sh[c:c+1, :], dim=1)

    irradiance = torch.clamp(irradiance, min=0.0)

    # BRDF已经在kernel中处理,直接相乘
    colors_precomp = diffuse_color * irradiance
    colors_precomp = torch.clamp(colors_precomp, min=0.0, max=1.0)

    # 渲染
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def create_env_sh(light_type="ambient", intensity=1.0, direction=None, color=None):
    """
    创建全局环境光的SH系数

    Returns:
        env_sh: (3, 9) tensor
    """
    env_sh = torch.zeros((3, 9), device="cuda")

    if color is None:
        color = torch.ones(3, device="cuda")
    else:
        color = torch.tensor(color, device="cuda", dtype=torch.float32)

    # SH基函数常量
    C0 = 0.282095  # Y_0,0
    C1 = 0.488603  # Y_1,m

    if light_type == "ambient":
        # 纯环境光 - 只有L0分量
        # 直接设置为归一化后的强度
        for c in range(3):
            env_sh[c, 0] = intensity * color[c] * C0

    elif light_type == "directional":
        # 方向光
        if direction is None:
            direction = np.array([0, 0, 1])
        direction = direction / np.linalg.norm(direction)
        dx, dy, dz = direction

        # L0: 基础环境光(小量)
        ambient_intensity = intensity * 0.2
        for c in range(3):
            env_sh[c, 0] = ambient_intensity * color[c] * C0

        # L1: 方向分量(主要能量)
        # 关键:不要再用RGB2SH!直接设置系数
        directional_intensity = intensity * 0.8  # 80%能量在方向上
        for c in range(3):
            env_sh[c, 1] = -C1 * dy * directional_intensity * color[c]
            env_sh[c, 2] = C1 * dz * directional_intensity * color[c]
            env_sh[c, 3] = -C1 * dx * directional_intensity * color[c]

    elif light_type == "sky":
        # 天空光照
        sky_color = torch.tensor([0.5, 0.7, 1.0], device="cuda") * intensity
        ground_color = torch.tensor([0.3, 0.3, 0.3], device="cuda") * intensity

        # L0: 平均颜色
        avg_color = (sky_color + ground_color) / 2
        for c in range(3):
            env_sh[c, 0] = avg_color[c] * C0

        # L1: Y_1,0 控制上下梯度
        diff_color = (sky_color - ground_color) * 0.5  # 减弱梯度
        for c in range(3):
            env_sh[c, 2] = C1 * diff_color[c]

    elif light_type == "sunset":
        # 日落
        sunset_color = torch.tensor([1.0, 0.5, 0.2], device="cuda") * intensity
        direction = np.array([1, 0, -0.3])
        direction = direction / np.linalg.norm(direction)
        dx, dy, dz = direction

        # L0: 环境
        for c in range(3):
            env_sh[c, 0] = sunset_color[c] * 0.3 * C0

        # L1: 方向
        directional_weight = 0.7
        for c in range(3):
            env_sh[c, 1] = -C1 * dy * sunset_color[c] * directional_weight
            env_sh[c, 2] = C1 * dz * sunset_color[c] * directional_weight
            env_sh[c, 3] = -C1 * dx * sunset_color[c] * directional_weight

    return env_sh


def create_env_sh_camera_light(camera, intensity=1.0):
    """
    创建相机位置的手电筒光照

    Args:
        camera: 相机对象
        intensity: 光照强度

    Returns:
        env_sh: (3, 9) tensor - 在世界坐标系下的环境光SH
    """
    # 获取相机位置和朝向
    # 相机朝向 = -Z轴方向(OpenGL convention)
    camera_transform = (
        camera.world_view_transform.inverse()
    )  # world_view 的逆是 camera_to_world

    # 提取光源方向(相机的-Z轴,即相机看向的方向)
    light_direction = -camera_transform[:3, 2].cpu().numpy()  # 第3列是Z轴
    light_direction = light_direction / np.linalg.norm(light_direction)

    dx, dy, dz = light_direction

    # 构建定向光SH
    env_sh = torch.zeros((3, 9), device="cuda")
    C1 = 0.488603

    # 小的环境光基底
    from utils.sh_utils import RGB2SH

    ambient_intensity = intensity * 0.1
    for c in range(3):
        env_sh[c, 0] = RGB2SH(torch.ones(3).cuda() * ambient_intensity)[c]

    # 强的定向光(白色)
    for c in range(3):
        env_sh[c, 1] = -C1 * dy * intensity
        env_sh[c, 2] = C1 * dz * intensity
        env_sh[c, 3] = -C1 * dx * intensity

    return env_sh


def render_with_camera_light(viewpoint_camera, pc, pipe, bg_color, intensity=2.0):
    """
    渲染带有相机手电筒效果的场景
    每一帧都动态计算相机方向的光照
    """
    # 动态创建环境光SH
    env_sh = create_env_sh_camera_light(viewpoint_camera, intensity)

    # 使用全局光照渲染
    return render_with_global_lighting(viewpoint_camera, pc, pipe, bg_color, env_sh)


def generate_camera_path(train_cameras, n_frames=240):
    """生成环绕相机轨迹"""
    from utils.render_utils import generate_path

    return generate_path(train_cameras, n_frames=n_frames)


def create_video_from_images(image_dir, output_path, fps=30):
    """从图像序列创建视频"""
    import cv2
    import glob

    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return

    # 读取第一张图片获取尺寸
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in tqdm(images, desc="Creating video"):
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"✓ Video saved to {output_path}")


def relight_scene_with_video(
    model_path,
    source_path,
    iteration,
    output_dir,
    lighting_configs,
    n_frames=240,
    fps=30,
):
    """
    使用全局光照进行重光照并生成视频
    """
    from argparse import Namespace

    gaussians = GaussianModel(3)

    dataset = Namespace(
        model_path=model_path,
        source_path=source_path,
        images="images",
        resolution=-1,
        white_background=True,
        data_device="cuda",
        eval=True,
        sh_degree=3,
        render_items=["RGB"],
    )

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )

    # 生成相机轨迹
    print(f"\nGenerating camera trajectory with {n_frames} frames...")
    train_cameras = scene.getTrainCameras()
    camera_path = generate_camera_path(train_cameras, n_frames=n_frames)

    for config_name, config in lighting_configs.items():
        print(f"\n{'='*60}")
        print(f"Rendering video: {config_name}")
        print(f"{'='*60}")

        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # === 特殊处理 flashlight ===
        if config["type"] == "flashlight":
            # 对每一帧使用相机位置的动态光照
            for idx, camera in enumerate(
                tqdm(camera_path, desc=f"Rendering {config_name}")
            ):
                with torch.no_grad():
                    render_pkg = render_with_camera_light(
                        camera,
                        gaussians,
                        pipe,
                        background,
                        intensity=config.get("intensity", 2.0),
                    )
                    rendering = render_pkg["render"]

                img = rendering.permute(1, 2, 0).cpu().numpy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(config_dir, f"{idx:05d}.png"))
        else:
            # 静态环境光
            env_sh = create_env_sh(
                light_type=config["type"],
                intensity=config.get("intensity", 1.0),
                direction=config.get("direction", None),
                color=config.get("color", None),
            )

            for idx, camera in enumerate(
                tqdm(camera_path, desc=f"Rendering {config_name}")
            ):
                with torch.no_grad():
                    render_pkg = render_with_global_lighting(
                        camera, gaussians, pipe, background, env_sh
                    )
                    rendering = render_pkg["render"]

                img = rendering.permute(1, 2, 0).cpu().numpy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(config_dir, f"{idx:05d}.png"))

        # 创建视频
        video_path = os.path.join(output_dir, f"{config_name}.mp4")
        create_video_from_images(config_dir, video_path, fps=fps)

        print(f"✓ Completed {config_name}")

    # 创建对比视频
    create_comparison_video(output_dir, lighting_configs, fps=fps)


def create_comparison_video(output_dir, lighting_configs, fps=30):
    """创建对比视频,并排显示多个光照配置"""
    import cv2
    import glob

    print(f"\n{'='*60}")
    print("Creating comparison video...")
    print(f"{'='*60}")

    config_names = list(lighting_configs.keys())

    # 选择要对比的配置(最多4个)
    selected_configs = config_names[: min(4, len(config_names))]

    # 读取第一帧获取尺寸
    first_dir = os.path.join(output_dir, selected_configs[0])
    images = sorted(glob.glob(os.path.join(first_dir, "*.png")))
    if len(images) == 0:
        return

    sample_img = cv2.imread(images[0])
    h, w, _ = sample_img.shape

    # 计算网格布局
    n_configs = len(selected_configs)
    if n_configs == 1:
        grid_h, grid_w = 1, 1
    elif n_configs == 2:
        grid_h, grid_w = 1, 2
    elif n_configs <= 4:
        grid_h, grid_w = 2, 2

    # 创建视频写入器
    video_w = w * grid_w
    video_h = h * grid_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "comparison.mp4")
    video = cv2.VideoWriter(video_path, fourcc, fps, (video_w, video_h))

    # 合成每一帧
    for frame_idx in tqdm(range(len(images)), desc="Comparison video"):
        canvas = np.zeros((video_h, video_w, 3), dtype=np.uint8)

        for idx, config_name in enumerate(selected_configs):
            img_path = os.path.join(output_dir, config_name, f"{frame_idx:05d}.png")
            img = cv2.imread(img_path)

            row = idx // grid_w
            col = idx % grid_w
            y_start = row * h
            x_start = col * w

            canvas[y_start : y_start + h, x_start : x_start + w] = img

            # 添加标签
            cv2.putText(
                canvas,
                config_name,
                (x_start + 10, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        video.write(canvas)

    video.release()
    print(f"✓ Comparison video saved to {video_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--source_path", "-s", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_dir", type=str, default="./relight_videos")
    parser.add_argument("--n_frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    lighting_configs = {
        "flashlight": {"type": "flashlight", "intensity": 3},  # 降低
        "bright_ambient": {"type": "ambient", "intensity": 3},  # 标准化为1
        "top_directional": {
            "type": "directional",
            "intensity": 3,  # 降低
            "direction": np.array([0, 0, 1]),
        },
        "sunset": {"type": "sunset", "intensity": 3},  # 大幅降低
    }

    relight_scene_with_video(
        args.model_path,
        args.source_path,
        args.iteration,
        args.output_dir,
        lighting_configs,
        n_frames=args.n_frames,
        fps=args.fps,
    )
