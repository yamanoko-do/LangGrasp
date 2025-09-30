"""
使用捕获的真实depth来优化MOGE估计的depth,包含缩放平移和旋转(基于ICP),随后投影回深度图中。
我保留了两个实现,另一个在depth_optimizer_discard.py中,测试下来精度无明显区别icp需要2s,而Umeyama需要8s,由于代码是gpt写的,我没有深扣细节，也许还有优化的空间
"""
import numpy as np
import cv2
import open3d as o3d
import os
from typing import Tuple, Optional
os.environ["XDG_SESSION_TYPE"] = "x11"

def calculate_optimal_scale(gt_depth: np.ndarray, pred_depth: np.ndarray) -> float:
    """
    计算预测深度图相对于真实深度图的最佳缩放因子
    
    Args:
        gt_depth: 真实深度图
        pred_depth: 预测深度图
        
    Returns:
        scale: 最佳缩放因子
    """
    mask = (gt_depth > 0) & (pred_depth > 0)   # 有效像素掩码
    gt = gt_depth[mask].reshape(-1)
    pred = pred_depth[mask].reshape(-1)
    
    # 最小二乘求因子 scale
    scale = np.dot(gt, pred) / np.dot(pred, pred)
    print(f"最佳缩放因子: {scale:.6f}")
    return scale

def icp_pointcloud_registration(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                               color: np.ndarray, intrinsic: np.ndarray,
                               voxel_size: float = 0.01) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    使用ICP进行点云配准
    
    Args:
        gt_depth: 真实深度图
        pred_depth: 预测深度图（已缩放）
        color: RGB颜色图像
        intrinsic: 相机内参矩阵
        voxel_size: 下采样体素大小
        
    Returns:
        transformation: 4x4变换矩阵
        pcd_pred_transformed: 变换后的预测点云
    """
    # 创建Open3D相机内参
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=gt_depth.shape[1],
        height=gt_depth.shape[0],
        fx=intrinsic[0,0],
        fy=intrinsic[1,1],
        cx=intrinsic[0,2],
        cy=intrinsic[1,2],
    )

    # 创建点云
    depth_o3d_gt = o3d.geometry.Image(gt_depth.astype(np.uint16))
    depth_o3d_pred = o3d.geometry.Image(pred_depth.astype(np.uint16))
    color_o3d = o3d.geometry.Image(color)

    rgbd_gt = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d_gt, depth_scale=1000.0, depth_trunc=5000.0, 
        convert_rgb_to_intensity=False)
    rgbd_pred = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d_pred, depth_scale=1000.0, depth_trunc=5000.0, 
        convert_rgb_to_intensity=False)

    pcd_gt = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_gt, intrinsic_o3d)
    pcd_pred = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_pred, intrinsic_o3d)

    # 下采样点云以提高ICP效率
    pcd_gt_down = pcd_gt.voxel_down_sample(voxel_size=voxel_size)
    pcd_pred_down = pcd_pred.voxel_down_sample(voxel_size=voxel_size)

    # 估计法线
    pcd_gt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_pred_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 执行ICP配准
    print("正在进行ICP配准...")
    threshold = 0.02  # 匹配距离阈值
    trans_init = np.eye(4)  # 初始变换矩阵

    # 点对点ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_pred_down, pcd_gt_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # 点对面ICP（通常更精确）
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd_pred_down, pcd_gt_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # 选择较好的结果
    if reg_p2l.fitness > reg_p2p.fitness:
        transformation = reg_p2l.transformation
        print("使用点对面ICP结果")
    else:
        transformation = reg_p2p.transformation
        print("使用点对点ICP结果")

    print("变换矩阵:")
    print(transformation)
    print(f"配准分数 (fitness): {max(reg_p2p.fitness, reg_p2l.fitness):.6f}")
    print(f"均方误差 (inlier_rmse): {min(reg_p2p.inlier_rmse, reg_p2l.inlier_rmse):.6f}")

    # 应用变换到原始预测点云
    pcd_pred_transformed = pcd_pred.transform(transformation)
    
    return transformation, pcd_pred_transformed

def pointcloud_to_depth(pcd: o3d.geometry.PointCloud, intrinsic: np.ndarray, 
                       width: int, height: int) -> np.ndarray:
    """
    将点云重新投影到深度图
    
    Args:
        pcd: 点云
        intrinsic: 相机内参矩阵
        width: 图像宽度
        height: 图像高度
        
    Returns:
        depth_map: 深度图
    """
    # 获取点云坐标
    points = np.asarray(pcd.points)
    
    # 过滤无效点
    valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
    points = points[valid_mask]
    
    if len(points) == 0:
        print("警告：没有有效的点云数据")
        return np.zeros((height, width), dtype=np.float32)
    
    # 投影到图像平面
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 相机坐标系到图像坐标系
    u = np.round(points[:, 0] * fx / points[:, 2] + cx).astype(int)
    v = np.round(points[:, 1] * fy / points[:, 2] + cy).astype(int)
    z = points[:, 2] * 1000.0  # 转换为毫米
    
    # 创建深度图
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # 过滤在图像范围内的点
    valid_pixels = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0)
    u_valid = u[valid_pixels]
    v_valid = v[valid_pixels]
    z_valid = z[valid_pixels]
    
    if len(u_valid) == 0:
        print("警告：没有有效的投影点")
        return depth_map
    
    # 使用最近邻填充
    from collections import defaultdict
    pixel_dict = defaultdict(list)
    
    for i in range(len(u_valid)):
        pixel_dict[(v_valid[i], u_valid[i])].append(z_valid[i])
    
    for (v, u), depths in pixel_dict.items():
        if 0 <= v < height and 0 <= u < width:
            depth_map[v, u] = min(depths)  # 取最近深度
    
    return depth_map

def fill_depth_holes(depth_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    填充深度图中的孔洞
    
    Args:
        depth_map: 输入深度图
        kernel_size: 形态学核大小
        
    Returns:
        filled_depth: 填充后的深度图
    """
    filled = depth_map.copy()
    mask = (depth_map > 0).astype(np.uint8)
    
    # 使用形态学操作填充小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 找到需要填充的孔洞
    holes = mask_closed - mask
    hole_coords = np.where(holes > 0)
    
    # 对每个孔洞像素，使用周围非零像素的平均值
    for i in range(len(hole_coords[0])):
        y, x = hole_coords[0][i], hole_coords[1][i]
        
        # 获取周围区域
        y_start = max(0, y - 1)
        y_end = min(depth_map.shape[0], y + 2)
        x_start = max(0, x - 1)
        x_end = min(depth_map.shape[1], x + 2)
        
        neighborhood = depth_map[y_start:y_end, x_start:x_end]
        non_zero = neighborhood[neighborhood > 0]
        
        if len(non_zero) > 0:
            filled[y, x] = np.mean(non_zero)
    
    return filled

def optimize_depth_map(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                      color_rgb: np.ndarray, intrinsic: np.ndarray,
                      voxel_size: float = 0.01) -> np.ndarray:
    """
    主函数：优化深度图
    
    Args:
        gt_depth: 真实深度图
        pred_depth: 预测深度图  
        color_rgb: RGB颜色图像
        intrinsic: 相机内参矩阵
        voxel_size: ICP下采样体素大小
        
    Returns:
        optimized_depth: 优化后的深度图 (uint16)
    """
    print("=== 开始深度图优化 ===")
    
    # 1. 计算缩放因子
    scale = calculate_optimal_scale(gt_depth, pred_depth)
    pred_depth_scaled = pred_depth * scale
    
    # 2. ICP点云配准
    transformation, pcd_pred_transformed = icp_pointcloud_registration(
        gt_depth, pred_depth_scaled, color_rgb, intrinsic, voxel_size)
    
    # 3. 点云重投影为深度图
    print("将点云重投影为深度图...")
    depth_optimized = pointcloud_to_depth(pcd_pred_transformed, intrinsic, 
                                         gt_depth.shape[1], gt_depth.shape[0])
    
    # 4. 孔洞填充
    depth_optimized_filled = fill_depth_holes(depth_optimized)
    
    # 5. 计算误差改善
    calculate_error_improvement(gt_depth, pred_depth_scaled, depth_optimized_filled)
    
    return depth_optimized_filled.astype(np.uint16)

def calculate_error_improvement(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                               optimized_depth: np.ndarray):
    """
    计算优化前后的误差对比
    """
    # 优化前误差
    mask_original = (gt_depth > 0) & (pred_depth > 0)
    if np.sum(mask_original) > 0:
        diff_original = pred_depth[mask_original] - gt_depth[mask_original]
        mae_original = np.mean(np.abs(diff_original))
        rmse_original = np.sqrt(np.mean(diff_original**2))
    else:
        mae_original = float('inf')
        rmse_original = float('inf')
    
    # 优化后误差
    mask_optimized = (gt_depth > 0) & (optimized_depth > 0)
    if np.sum(mask_optimized) > 0:
        diff_optimized = optimized_depth[mask_optimized] - gt_depth[mask_optimized]
        mae_optimized = np.mean(np.abs(diff_optimized))
        rmse_optimized = np.sqrt(np.mean(diff_optimized**2))
        
        improvement_mae = mae_original - mae_optimized
        improvement_rmse = rmse_original - rmse_optimized
    else:
        mae_optimized = float('inf')
        rmse_optimized = float('inf')
        improvement_mae = 0
        improvement_rmse = 0
    
    print("\n=== 误差分析 ===")
    print("优化前 - MAE: {:.4f}, RMSE: {:.4f}".format(mae_original, rmse_original))
    print("优化后 - MAE: {:.4f}, RMSE: {:.4f}".format(mae_optimized, rmse_optimized))
    print("改善量 - MAE: {:.4f}, RMSE: {:.4f}".format(improvement_mae, improvement_rmse))

def visualize_depth_comparison(color_rgb: np.ndarray, gt_depth: np.ndarray, 
                              pred_depth: np.ndarray, optimized_depth: np.ndarray,
                              intrinsic: np.ndarray):
    """
    可视化对比真实深度、预测深度和优化后的深度
    
    Args:
        color_rgb: RGB颜色图像
        gt_depth: 真实深度图
        pred_depth: 预测深度图
        optimized_depth: 优化后的深度图
        intrinsic: 相机内参矩阵
    """
    print("\n=== 创建可视化 ===")
    
    # 创建Open3D相机内参
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=gt_depth.shape[1],
        height=gt_depth.shape[0],
        fx=intrinsic[0,0],
        fy=intrinsic[1,1],
        cx=intrinsic[0,2],
        cy=intrinsic[1,2],
    )

    # 创建点云
    color_o3d = o3d.geometry.Image(color_rgb)
    
    depth_gt_o3d = o3d.geometry.Image(gt_depth.astype(np.uint16))
    depth_pred_o3d = o3d.geometry.Image(pred_depth.astype(np.uint16))
    depth_optimized_o3d = o3d.geometry.Image(optimized_depth.astype(np.uint16))

    rgbd_gt = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_gt_o3d, depth_scale=1000.0, depth_trunc=5000.0, 
        convert_rgb_to_intensity=False)
    rgbd_pred = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_pred_o3d, depth_scale=1000.0, depth_trunc=5000.0, 
        convert_rgb_to_intensity=False)
    rgbd_optimized = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_optimized_o3d, depth_scale=1000.0, depth_trunc=5000.0, 
        convert_rgb_to_intensity=False)

    pcd_gt = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_gt, intrinsic_o3d)
    pcd_pred = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_pred, intrinsic_o3d)
    pcd_optimized = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_optimized, intrinsic_o3d)

    def tint_pointcloud(pcd, tint_color, alpha=0.3):
        """给点云着色"""
        colors = np.asarray(pcd.colors)
        if len(colors) == 0:
            points = np.asarray(pcd.points)
            if len(points) > 0:
                colors = np.ones_like(points) * 0.5
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                return pcd
        
        tint = np.array(tint_color).reshape(1, 3)
        colors_tinted = (1 - alpha) * colors + alpha * tint
        pcd.colors = o3d.utility.Vector3dVector(colors_tinted)
        return pcd

    # 给点云着色
    pcd_gt_tinted = tint_pointcloud(pcd_gt, [0.0, 0.0, 1.0], alpha=0.3)   # 蓝色：真实点云
    pcd_pred_tinted = tint_pointcloud(pcd_pred, [1.0, 0.0, 0.0], alpha=0.3)  # 红色：原始预测
    pcd_optimized_tinted = tint_pointcloud(pcd_optimized, [0.0, 1.0, 0.0], alpha=0.3)  # 绿色：优化后

    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="深度图优化可视化")

    # 初始显示所有点云
    for pcd in [pcd_gt_tinted, pcd_pred_tinted, pcd_optimized_tinted]:
        vis.add_geometry(pcd)

    def switch_geometry(vis, pcds):
        """切换显示的几何体"""
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()

        vis.clear_geometries()
        for p in pcds:
            vis.add_geometry(p)

        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        return False

    # 定义快捷键回调函数
    def show_gt(vis):
        return switch_geometry(vis, [pcd_gt_tinted])

    def show_pred(vis):
        return switch_geometry(vis, [pcd_pred_tinted])

    def show_optimized(vis):
        return switch_geometry(vis, [pcd_optimized_tinted])

    def show_gt_vs_pred(vis):
        return switch_geometry(vis, [pcd_gt_tinted, pcd_pred_tinted])

    def show_gt_vs_optimized(vis):
        return switch_geometry(vis, [pcd_gt_tinted, pcd_optimized_tinted])

    def show_all(vis):
        return switch_geometry(vis, [pcd_gt_tinted, pcd_pred_tinted, pcd_optimized_tinted])

    # 绑定快捷键
    vis.register_key_callback(ord("1"), show_gt)              # 1: 只显示真实点云
    vis.register_key_callback(ord("2"), show_pred)            # 2: 只显示原始预测
    vis.register_key_callback(ord("3"), show_optimized)       # 3: 只显示优化后
    vis.register_key_callback(ord("4"), show_gt_vs_pred)      # 4: 真实 vs 原始预测
    vis.register_key_callback(ord("5"), show_gt_vs_optimized) # 5: 真实 vs 优化后
    vis.register_key_callback(ord("6"), show_all)             # 6: 显示所有

    print("\n=== 可视化控制 ===")
    print("1: 真实点云 (蓝色)")
    print("2: 原始预测 (红色)") 
    print("3: 优化后 (绿色)")
    print("4: 真实 vs 原始预测")
    print("5: 真实 vs 优化后")
    print("6: 显示所有")
    print("Q: 退出可视化")

    vis.run()
    vis.destroy_window()



if __name__ == "__main__":
    os.environ["XDG_SESSION_TYPE"] = "x11"
    # 数据路径
    data_dir = "/root/host_share/LangGrasp/data/"
    
    # 相机内参
    intrinsic = np.array([
        [896.4105731015486, 0.0, 646.5148650122112],
        [0.0, 895.8013693817877, 374.5274283245406],
        [0.0, 0.0, 1.0]
    ])
    
    # 读取图像数据
    print("读取图像数据...")
    depth_gt = cv2.imread(data_dir + "depth_measure.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_pred = cv2.imread(data_dir + "depth_infer.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    color_bgr = cv2.imread(data_dir + "color.png")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"深度图尺寸: {depth_gt.shape}")
    print(f"颜色图尺寸: {color_rgb.shape}")
    
    # 优化深度图
    import time
    start = time.time()
    depth_optimized = optimize_depth_map(depth_gt, depth_pred, color_rgb, intrinsic)
    end = time.time()
    print(f"耗时：{end - start:.4f} 秒")
    # 保存优化后的深度图
    output_path = data_dir + "depth_optimized.png"
    cv2.imwrite(output_path, depth_optimized)
    print(f"\n优化后的深度图已保存到: {output_path}")
    
    # 计算缩放后的预测深度（用于可视化对比）
    scale = calculate_optimal_scale(depth_gt, depth_pred)
    depth_pred_scaled = (depth_pred * scale).astype(np.uint16)
    
    # 可视化对比
    visualize_depth_comparison(color_rgb, depth_gt.astype(np.uint16), 
                              depth_pred_scaled, depth_optimized, intrinsic)