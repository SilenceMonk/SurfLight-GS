import torch
import os
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from arguments import ModelParams
import numpy as np
from PIL import Image
from tqdm import tqdm
import math


def render_material_channels(viewpoint_camera, pc, pipe, bg_color, channel="albedo"):
    """
    渲染材质通道(albedo, normals等)

    Args:
        channel: 'albedo', 'normal', 'geometry'
    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from utils.general_utils import build_rotation

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

    # 根据通道选择要渲染的颜色
    if channel == "albedo":
        colors_precomp = pc.get_diffuse_color

    elif channel == "normal":
        R = build_rotation(pc._rotation)
        normals = R[:, :, 2]
        normals = torch.nn.functional.normalize(normals, dim=-1)
        colors_precomp = normals * 0.5 + 0.5

    elif channel == "geometry":
        colors_precomp = torch.ones_like(pc.get_xyz) * 0.5

    else:
        raise ValueError(f"Unknown channel: {channel}")

    colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)

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

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in tqdm(images, desc="Creating video"):
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"✓ Video saved to {output_path}")


def visualize_materials_video(
    model_path, source_path, iteration, output_dir, n_frames=240, fps=30
):
    """
    渲染材质通道视频
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

    channels = {
        "albedo": "Diffuse Color (Material)",
        "normal": "Surface Normals",
        "geometry": "Geometry Only",
    }

    # 渲染每个通道
    for channel_name, description in channels.items():
        print(f"\n{'='*60}")
        print(f"Rendering {description}")
        print(f"{'='*60}")

        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        for idx, camera in enumerate(tqdm(camera_path, desc=channel_name)):
            with torch.no_grad():
                render_pkg = render_material_channels(
                    camera, gaussians, pipe, background, channel=channel_name
                )
                rendering = render_pkg["render"]

            img = rendering.permute(1, 2, 0).cpu().numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(channel_dir, f"{idx:05d}.png"))

        # 创建视频
        video_path = os.path.join(output_dir, f"{channel_name}.mp4")
        create_video_from_images(channel_dir, video_path, fps=fps)
        print(f"✓ Completed {channel_name}")

    # 创建对比视频
    create_comparison_video(output_dir, channels, n_frames, fps)

    print(f"\n{'='*60}")
    print(f"All videos saved to {output_dir}")
    print(f"{'='*60}")


def create_comparison_video(output_dir, channels, n_frames, fps=30):
    """创建并排对比视频"""
    import cv2

    channel_names = list(channels.keys())

    # 读取第一帧获取尺寸
    first_img = cv2.imread(os.path.join(output_dir, channel_names[0], "00000.png"))
    h, w, _ = first_img.shape

    # 水平排列
    video_w = w * len(channel_names)
    video_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "comparison.mp4")
    video = cv2.VideoWriter(video_path, fourcc, fps, (video_w, video_h))

    print(f"\nCreating comparison video...")
    for frame_idx in tqdm(range(n_frames), desc="Comparison"):
        canvas = np.zeros((video_h, video_w, 3), dtype=np.uint8)

        for col_idx, channel_name in enumerate(channel_names):
            img_path = os.path.join(output_dir, channel_name, f"{frame_idx:05d}.png")
            img = cv2.imread(img_path)

            x_start = col_idx * w
            canvas[:, x_start : x_start + w] = img

            # 添加标签
            cv2.putText(
                canvas,
                channel_name.upper(),
                (x_start + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        video.write(canvas)

    video.release()
    print(f"✓ Comparison video saved to {video_path}")


def visualize_materials_static(model_path, source_path, iteration, output_dir):
    """静态图像版本(原功能)"""
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

    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        test_cameras = scene.getTrainCameras()[:5]

    channels = {
        "albedo": "Diffuse Color",
        "normal": "Surface Normals",
        "geometry": "Geometry Only",
    }

    for channel_name, description in channels.items():
        print(f"\nRendering {description}...")
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        for idx, camera in enumerate(tqdm(test_cameras, desc=channel_name)):
            with torch.no_grad():
                render_pkg = render_material_channels(
                    camera, gaussians, pipe, background, channel=channel_name
                )
                rendering = render_pkg["render"]

            img = rendering.permute(1, 2, 0).cpu().numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(channel_dir, f"view_{idx:03d}.png"))

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--source_path", "-s", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_dir", type=str, default="./material_vis")
    parser.add_argument(
        "--video", action="store_true", help="Render video instead of static images"
    )
    parser.add_argument("--n_frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.video:
        visualize_materials_video(
            args.model_path,
            args.source_path,
            args.iteration,
            args.output_dir,
            n_frames=args.n_frames,
            fps=args.fps,
        )
    else:
        visualize_materials_static(
            args.model_path, args.source_path, args.iteration, args.output_dir
        )
