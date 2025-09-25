"""
使用捕获的真实depth来优化MOGE估计的depth,包含缩放平移和旋转(基于Umeyama)，随后投影回深度图中
"""

import numpy as np
import cv2
import open3d as o3d
import os


# --------------------
# 工具函数
# --------------------
def depth_to_points(depth_mm, intrinsic):
    """depth_mm: HxW depth in millimeters (float32)。
       返回 Nx3 的点阵（单位 m）和对应的像素坐标 Nx2"""
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    mask = (depth_mm > 0)
    ys, xs = np.nonzero(mask)
    zs = depth_mm[ys, xs] / 1000.0  # mm->m
    xs_3d = (xs - cx) * zs / fx
    ys_3d = (ys - cy) * zs / fy
    pts = np.stack([xs_3d, ys_3d, zs], axis=1)
    pix = np.stack([xs, ys], axis=1)
    return pts, pix, mask


def umeyama(src, dst, with_scaling=True):
    """Umeyama 相似变换估计"""
    assert src.shape == dst.shape
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov = (dst_centered.T @ src_centered) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2,2] = -1

    R = U @ S @ Vt
    if with_scaling:
        var_src = np.sum(src_centered**2) / n
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mean_dst - s * R @ mean_src
    return s, R, t


# --------------------
# 误差计算函数
# --------------------
def calculate_error_improvement(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                               optimized_depth: np.ndarray):
    """
    计算优化前后的误差对比 (MAE / RMSE)
    """
    # 优化前
    mask_original = (gt_depth > 0) & (pred_depth > 0)
    if np.sum(mask_original) > 0:
        diff_original = pred_depth[mask_original] - gt_depth[mask_original]
        mae_original = np.mean(np.abs(diff_original))
        rmse_original = np.sqrt(np.mean(diff_original**2))
    else:
        mae_original = float("inf")
        rmse_original = float("inf")

    # 优化后
    mask_optimized = (gt_depth > 0) & (optimized_depth > 0)
    if np.sum(mask_optimized) > 0:
        diff_optimized = optimized_depth[mask_optimized] - gt_depth[mask_optimized]
        mae_optimized = np.mean(np.abs(diff_optimized))
        rmse_optimized = np.sqrt(np.mean(diff_optimized**2))

        improvement_mae = mae_original - mae_optimized
        improvement_rmse = rmse_original - rmse_optimized
    else:
        mae_optimized = float("inf")
        rmse_optimized = float("inf")
        improvement_mae = 0
        improvement_rmse = 0

    print("\n=== 误差分析 ===")
    print("优化前 - MAE: {:.4f}, RMSE: {:.4f}".format(mae_original, rmse_original))
    print("优化后 - MAE: {:.4f}, RMSE: {:.4f}".format(mae_optimized, rmse_optimized))
    print("改善量 - MAE: {:.4f}, RMSE: {:.4f}".format(improvement_mae, improvement_rmse))


# --------------------
# 主处理函数
# --------------------
def optimize_depth(depth_gt, depth_pred, intrinsic):
    """
    输入:
        depth_gt: HxW, float32 (mm)
        depth_pred: HxW, float32 (mm)
        intrinsic: 3x3 相机内参
    返回:
        depth_optimized: HxW, uint16 (mm)
    """
    h, w = depth_gt.shape
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]

    # 公共 mask
    common_mask = (depth_gt > 0) & (depth_pred > 0)
    ys, xs = np.nonzero(common_mask)
    if len(xs) < 6:
        raise RuntimeError("有效的对应像素太少，无法做刚性/相似变换估计。")

    pts_gt_common = np.stack([
        (xs - cx) * (depth_gt[ys, xs] / 1000.0) / fx,
        (ys - cy) * (depth_gt[ys, xs] / 1000.0) / fy,
        (depth_gt[ys, xs] / 1000.0)
    ], axis=1)

    pts_pred_common = np.stack([
        (xs - cx) * (depth_pred[ys, xs] / 1000.0) / fx,
        (ys - cy) * (depth_pred[ys, xs] / 1000.0) / fy,
        (depth_pred[ys, xs] / 1000.0)
    ], axis=1)

    # Umeyama
    s, R, t = umeyama(pts_pred_common, pts_gt_common, with_scaling=True)
    print("估计的 scale:", s)

    # 全部预测点
    pts_pred_all, _, _ = depth_to_points(depth_pred, intrinsic)
    pts_pred_all_transformed = (s * (R @ pts_pred_all.T)).T + t

    # 投影回像素平面（z-buffer）
    depth_optimized = np.zeros((h, w), dtype=np.uint16)
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)

    xs_3d, ys_3d, zs_3d = pts_pred_all_transformed[:, 0], pts_pred_all_transformed[:, 1], pts_pred_all_transformed[:, 2]
    us = (xs_3d * fx) / zs_3d + cx
    vs = (ys_3d * fy) / zs_3d + cy
    us_i = np.round(us).astype(np.int32)
    vs_i = np.round(vs).astype(np.int32)

    valid = (zs_3d > 0) & (us_i >= 0) & (us_i < w) & (vs_i >= 0) & (vs_i < h)
    for u, v, z in zip(us_i[valid], vs_i[valid], zs_3d[valid]):
        if z < z_buffer[v, u]:
            z_buffer[v, u] = z
            mm = int(np.round(z * 1000.0))
            depth_optimized[v, u] = np.clip(mm, 0, 65535)

    # 误差分析
    calculate_error_improvement(depth_gt, depth_pred, depth_optimized)

    return depth_optimized.astype(np.uint16)


# --------------------
# 可视化函数
# --------------------
def visualize_depths(color_rgb, depth_gt, depth_pred, depth_opt, intrinsic):
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    h, w = depth_gt.shape

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    color_o3d = o3d.geometry.Image(color_rgb)

    def to_pcd(depth_map):
        depth_o3d = o3d.geometry.Image(depth_map)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0, depth_trunc=5000.0,
            convert_rgb_to_intensity=False
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)

    pcd_gt = to_pcd(depth_gt)
    pcd_pred = to_pcd(depth_pred)
    pcd_opt = to_pcd(depth_opt)

    def tint(pcd, color, alpha=0.3):
        if not pcd.has_colors():
            pcd.paint_uniform_color([0.5,0.5,0.5])
        colors = np.asarray(pcd.colors)
        tint = np.array(color).reshape(1,3)
        colors_tinted = (1-alpha)*colors + alpha*tint
        pcd.colors = o3d.utility.Vector3dVector(colors_tinted)
        return pcd

    pcd_gt = tint(pcd_gt, [0,0,1], 0.3)       # 蓝 GT
    pcd_pred = tint(pcd_pred, [1,0.5,0], 0.35)# 橙 预测
    pcd_opt = tint(pcd_opt, [1,0,0], 0.3)     # 红 优化

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    for p in [pcd_gt, pcd_opt]:
        vis.add_geometry(p)

    def switch(vis, pcds):
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        for p in pcds:
            vis.add_geometry(p)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        return False

    vis.register_key_callback(ord("1"), lambda v: switch(v, [pcd_gt]))
    vis.register_key_callback(ord("2"), lambda v: switch(v, [pcd_pred]))
    vis.register_key_callback(ord("3"), lambda v: switch(v, [pcd_opt]))
    vis.register_key_callback(ord("4"), lambda v: switch(v, [pcd_gt, pcd_opt]))

    print("按 1: GT, 2: 预测, 3: 优化, 4: GT+优化")
    vis.run()
    vis.destroy_window()


# --------------------
# main
# --------------------
if __name__ == "__main__":
    os.environ["XDG_SESSION_TYPE"] = "x11"
    data_dir = "/root/host_share/langrasp/data/"

    depth = cv2.imread(os.path.join(data_dir, "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_pred = cv2.imread(os.path.join(data_dir, "depth_infer.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    color_bgr = cv2.imread(os.path.join(data_dir, "color.png"))
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    intrinsic = np.array([
        [896.4105731015486, 0.0, 646.5148650122112],
        [0.0, 895.8013693817877, 374.5274283245406],
        [0.0, 0.0, 1.0]
    ])
    import time
    start = time.time()
    depth_opt = optimize_depth(depth, depth_pred, intrinsic)
    end = time.time()
    print(f"耗时：{end - start:.4f} 秒")
    cv2.imwrite(os.path.join(data_dir, "depth_infer_optimized.png"), depth_opt)

    visualize_depths(color_rgb, depth.astype(np.uint16), depth_pred.astype(np.uint16), depth_opt, intrinsic)
