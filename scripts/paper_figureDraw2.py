import os
import cv2
import numpy as np
from langgrasp.depth_optimizer import optimize_depth_map, calculate_optimal_scale, visualize_depth_comparison
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
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