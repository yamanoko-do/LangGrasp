import os
import cv2
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def show_image(image: np.ndarray, title: str = "Image", show_colorbar: bool = None, figsize=(6, 6)):
    """
    自动判断图像类型（RGB 彩色图 or 单通道深度图）并可视化

    参数:
        image (np.ndarray): 输入图像数组
            - 彩色图: (H, W, 3) 或 (H, W, 4)
            - 深度图: (H, W) 或 (H, W, 1)
        title (str): 图像标题
        show_colorbar (bool): 是否显示 colorbar（深度图默认 True，彩色图默认 False）
        figsize (tuple): 图像大小

    示例:
        visualize_image_auto(color_img)      # 自动识别为彩色图
        visualize_image_auto(depth_img)      # 自动识别为深度图
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("输入必须是 numpy.ndarray")
    print(f"图像形状: {image.shape}, 数据类型: {image.dtype}")
    # 处理 (H, W, 1) -> (H, W)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)  # 变成 (H, W)

    # 判断图像类型
    is_color = (image.ndim == 3 and image.shape[2] in [3, 4])
    is_depth = (image.ndim == 2)

    if not (is_color or is_depth):
        raise ValueError(f"不支持的图像形状: {image.shape}。仅支持 (H, W, 3/4) 彩色图 或 (H, W) 深度图。")

    # 值域归一化处理（仅对彩色图需要，深度图 matplotlib 会自动映射）
    img_display = image.astype(np.float32)
    if is_color:
        if img_display.max() > 1.0:
            img_display = img_display / 255.0
        img_display = np.clip(img_display, 0.0, 1.0)

    # 设置默认 colorbar 行为
    if show_colorbar is None:
        show_colorbar = is_depth  # 深度图默认显示 colorbar

    # 绘图
    plt.figure(figsize=figsize)
    if is_color:
        plt.imshow(img_display)
    else:  # 深度图
        im = plt.imshow(img_display, cmap='jet')
        if show_colorbar:
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Depth Value')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_pointcloud_from_rgbd(intrinsic, color_img, depth_img, mask_img=None,
                                near=0.1, far=5.0, is_depth_buffer=False,
                                visualize=True, save_path=None):
    """
    从RGB-D图像创建并可视化点云
    会使用内置的“保存的相机参数”来初始化视角，并实时打印当前视角参数
    """
    # 确保环境变量设置（X11）
    os.environ["XDG_SESSION_TYPE"] = "x11"

    height, width = depth_img.shape[:2]

    # 深度值处理
    if is_depth_buffer:
        depth_real = far * near / (far - (far - near) * depth_img)
    else:
        depth_real = depth_img

    # 生成网格坐标
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    depth_flat = depth_real.reshape(-1)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x = (xs - cx) * depth_flat / fx
    y = (ys - cy) * depth_flat / fy
    z = depth_flat

    points = np.stack((x, y, z), axis=1)
    colors = color_img.reshape(-1, 3) / 255.0

    valid = (z > 0)
    if mask_img is not None:
        mask_flat = mask_img.reshape(-1)
        if mask_img.dtype == bool:
            valid = valid & mask_flat
        elif mask_img.dtype == np.uint8:
            valid = valid & (mask_flat > 127)
    else:
        mask_img = np.ones((height, width), dtype=bool)

    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存点云（如果需要）
    # if save_path:
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
    #     print(f"点云已保存至: {save_path}")

    if visualize:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # -------------- 这里定义你想要恢复的“已保存相机参数”（保持在函数内） --------------
        saved_intrinsic_matrix = np.array([[935.3074360871938, 0.0, 959.5], [0.0, 935.3074360871938, 539.5], [0.0, 0.0, 1.0]])
        saved_extrinsic = np.array([[-0.8740589989516001, 0.48234825439395235, 0.057974372224576826, -1.5567185844296034], [-0.23997040003487355, -0.532414827512258, 0.8117565266458875, -698.9678025988629], [0.42241575900862127, 0.69561096377803, 0.5811112747253927, 500.3704540518654], [0.0, 0.0, 0.0, 1.0]])
        # -------------------------------------------------------------------------------

        # 从 saved_intrinsic_matrix 推断保存时的宽高（Open3D 使用的像素中心定义为 (width-1)/2）
        sx_cx = float(saved_intrinsic_matrix[0, 2])
        sy_cy = float(saved_intrinsic_matrix[1, 2])
        sx_fx = float(saved_intrinsic_matrix[0, 0])
        sy_fy = float(saved_intrinsic_matrix[1, 1])

        saved_w = int(round(2.0 * sx_cx + 1.0))
        saved_h = int(round(2.0 * sy_cy + 1.0))

        # 创建可视化器并指定窗口大小为保存时的分辨率，避免 convert 时尺寸不匹配
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=saved_w, height=saved_h)
        vis.add_geometry(pcd)
        vis.add_geometry(coordinate_frame)

        # 构造 PinholeCameraParameters（注意使用 PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)）
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(saved_w, saved_h,
                                                            sx_fx, sy_fy, sx_cx, sy_cy)
        param.extrinsic = saved_extrinsic

        # 尝试应用，如果失败则回退到按窗口缩放 intrinsic 的策略
        ctr = vis.get_view_control()
        try:
            ctr.convert_from_pinhole_camera_parameters(param)
            applied_by = "直接按保存的 (width,height,fx,fy,cx,cy) 应用"
        except Exception as e:
            # 这里给出提示并尝试缩放到当前窗口尺寸
            print("[WARN] convert_from_pinhole_camera_parameters 失败，尝试按窗口尺寸缩放 intrinsic。")
            # 获取实际创建的窗口尺寸（我们在 create_window 时指定了 saved_w/saved_h，通常一致）
            # 但为稳妥起见，我们以 saved_w/saved_h 为目标（如果你想以其它尺寸启动窗口，可改这里）
            cur_w, cur_h = saved_w, saved_h

            # 计算缩放因子（如果 cur 与 saved 不同）
            scale_x = cur_w / float(saved_w)
            scale_y = cur_h / float(saved_h)

            fx_scaled = sx_fx * scale_x
            fy_scaled = sy_fy * scale_y
            cx_scaled = sx_cx * scale_x
            cy_scaled = sy_cy * scale_y

            # 重新构造 intrinsic 并应用
            param.intrinsic = o3d.camera.PinholeCameraIntrinsic(cur_w, cur_h,
                                                                fx_scaled, fy_scaled,
                                                                cx_scaled, cy_scaled)
            try:
                ctr.convert_from_pinhole_camera_parameters(param)
                applied_by = "按窗口缩放后应用"
            except Exception as e2:
                print("[ERROR] 仍然无法应用相机参数（convert_from_pinhole_camera_parameters）。继续打开可视化但不初始化视角。")
                applied_by = "未成功应用（保持默认视角）"

        # 实时打印相机参数（每秒一次）
        last_print_time = time.time()

        def print_view_parameters(vis):
            nonlocal last_print_time
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                try:
                    cur_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                    intrinsic_mat = cur_param.intrinsic.intrinsic_matrix
                    extrinsic_mat = cur_param.extrinsic
                    print("\n当前相机参数:")
                    print(f"Intrinsic:\n{intrinsic_mat.tolist()}")
                    print(f"Extrinsic:\n{extrinsic_mat.tolist()}")
                    print(f"(初始化应用方式: {applied_by})")
                except Exception as e:
                    # 在某些 Open3D 版本中 convert_to_pinhole_camera_parameters 也可能失败，
                    # 这里捕获并打印简单信息，避免崩溃
                    print("[WARN] 无法从当前视图获取 PinholeCameraParameters：", e)
                last_print_time = current_time
            return False

        vis.register_animation_callback(print_view_parameters)

        print("可视化窗口已打开，正在实时打印视角参数...")
        print("调整视角后，这些参数可用于设置初始视角（如需保存请记录上面打印的 intrinsic/extrinsic）")
        print("按 ESC 关闭窗口")

        vis.run()
        vis.destroy_window()
    return pcd

if __name__ == "__main__":
    """
    create_pointcloud_from_rgbd
    """
    # 相机内参 (要和渲染时保持一致)
    # width, height = 640, 480
    # fov = 60
    # aspect = width / height
    # near = 0.1  # 近裁切平面
    # far = 5     # 远裁切平面

    # f = height / (2 * np.tan(fov * np.pi / 360))

    # intrinsic = np.array([[f, 0, width / 2],
    #                     [0, f, height / 2],
    #                     [0, 0, 1]])
    intrinsic = np.array([[606, 0, 327],
                        [0, 606, 247],
                        [0, 0, 1]])
    
    # 加载图像数据 (替换为实际路径)
    color_img = cv2.cvtColor(cv2.imread('data/color_gg.png'), cv2.COLOR_BGR2RGB)
    #depth_img = np.load('data/depth.npy')
    depth_img = cv2.imread("data/depth_gg.png", cv2.IMREAD_ANYDEPTH) 
    #mask_img = cv2.imread('data/mask.png', cv2.IMREAD_GRAYSCALE)
    print(color_img.dtype)
    print(depth_img.dtype)
    #print(mask_img.dtype)
    print(type(color_img))
    print(type(depth_img))
    #print(type(mask_img))

    # 创建并可视化点云 (根据深度图类型设置is_depth_buffer)
    pcd=create_pointcloud_from_rgbd(
            intrinsic=intrinsic,
            color_img=color_img,
            depth_img=depth_img,
            #mask_img=mask_img,
            near=0.1,
            far=5.0,
            is_depth_buffer=False,  # 根据实际情况设置为True或False
        )