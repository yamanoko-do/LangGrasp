"""
抓取生成类,是对graspnet的包装
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup



from graspnetbaseline.models.graspnet import GraspNet, pred_decode
from graspnetbaseline.dataset.graspnet_dataset import GraspNetDataset
from graspnetbaseline.utils.collision_detector import ModelFreeCollisionDetector
from graspnetbaseline.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

"""
初始化抓取网络
Args:
    num_view (int): 生成抓取姿态时，采样的抓取视角数目，过少可能会漏掉最优抓取方向
    b (int): 除数。
Returns:
    tuple: 包含两个元素的元组，分别是：
        - quotient (int): 商，表示 a 除以 b 的整数结果。
        - remainder (int): 余数，表示 a 除以 b 的余数。
"""

def get_net(checkpoint_path):
    """
    初始化抓取网络
    """
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def get_and_process_data(color, depth, cam_info, target_mask=None):
    """
    color: (H, W, 3) 0-255
    depth: (H, W)
    intrinsic: 3x3
    target_mask: (H, W) 布尔数组 or None,表示目标区域
    返回:
        end_points: 供网络使用（基于全场景采样）
        cloud_o3d: 全场景点云,用于碰撞检测和可视化
        object_pc: 目标点云,用来过滤抓取
    """
    num_point = 20000
    color = np.array(color, dtype=np.float32) / 255.0
    factor_depth = 1000.0
    intrinsic = cam_info["intrinsic"]
    camera = CameraInfo(*cam_info["cam_rgb_hw"],
                        intrinsic[0][0], intrinsic[1][1],
                        intrinsic[0][2], intrinsic[1][2],
                        factor_depth)

    organized_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 过滤有效的点
    valid_mask = (depth > 0)
    # 使用目标mask作为工作空间
    # valid_mask = (depth > 0) & (target_mask>0)

    # scene point cloud (numpy Nx3) -- 用于网络与可视化与碰撞检测
    scene_pc = organized_cloud[valid_mask]
    scene_color = color[valid_mask]

    # object_pc 只从 target_mask + valid_mask 中提取，用于后续过滤
    object_pc = np.zeros((0, 3), dtype=np.float32)
    if target_mask is not None:
        # target_mask 应与 depth 尺寸一致 (H, W)
        object_mask = valid_mask & target_mask
        object_pc = organized_cloud[object_mask].astype(np.float32)

    # 场景点云下采样
    if len(scene_pc) >= num_point:
        idxs = np.random.choice(len(scene_pc), num_point, replace=False)
    else:
        idxs1 = np.arange(len(scene_pc))
        idxs2 = np.random.choice(len(scene_pc), num_point - len(scene_pc), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = scene_pc[idxs]
    color_sampled = scene_color[idxs]

    # 构造 open3d 点云（用于可视化与碰撞检测）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(scene_pc.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(scene_color.astype(np.float32))

    end_points = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled_t = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points['point_clouds'] = cloud_sampled_t
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d, object_pc


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud,collision_thresh,voxel_size):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud, view_num = 50 ,window_name = "Grasp Visualization"):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:view_num]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers],window_name)


def filter_grasps_by_mask(gg: GraspGroup, object_pc: np.ndarray, thresh=0.01):
    from sklearn.neighbors import KDTree
    """
    使用 mask 点云过滤 grasps
    gg: GraspGroup
    object_pc: (N, 3) 目标点云 (由mask提取)
    thresh: 抓取中心到mask点云的最大距离,米
    """
    tree = KDTree(object_pc)
    keep_ids = []
    for i, g in enumerate(gg):
        center = g.translation  # 抓取中心位置
        dist, _ = tree.query([center], k=1)
        if dist[0][0] < thresh:  # 在mask点云附近
            keep_ids.append(i)
    return gg[keep_ids]

def get_grasp(net, scence_image,scence_depth, target_mask, cam_info):
    """
    根据场景图像rgb,depth和目标mask返回最优抓取
    """
    # 如果 mask 是 (H, W, 3) 的 numpy 数组
    if target_mask.ndim == 3:
        # 常见的mask是黑白图，RGB三个通道值相同，取一个通道即可
        target_mask = target_mask[:, :, 0]

    # 转成布尔数组（假设前景是 >0 的值）
    target_mask = target_mask > 0

    collision_thresh = 0.01 #碰撞检测阈值，default0.01
    voxel_size = 0.01 #在碰撞检测前处理点云的体素大小，default0.01
    
    end_points, cloud ,object_pc= get_and_process_data(scence_image, scence_depth, cam_info, target_mask)
    gg = get_grasps(net, end_points)
    #breakpoint()
    if collision_thresh:
        gg = collision_detection(gg, np.array(cloud.points),collision_thresh, voxel_size)
    vis_grasps(gg = gg, cloud = cloud, view_num = 50, window_name = "all grasp poses")
    gg = filter_grasps_by_mask(gg,object_pc,0.05)
    #vis_grasps(gg = gg, cloud = cloud, view_num = 1, window_name = "bestpose")
    #sortgg = gg.sort_by_score()
    return gg ,cloud

