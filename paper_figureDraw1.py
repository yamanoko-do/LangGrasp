import os
import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from ultralytics import FastSAM, SAM
from langrasp.config import Config
from langrasp.camera import CameraD435
from langrasp.piper_obj import PiperClass
from langrasp.mask import get_target_mask,save_mask_as_image
from langrasp.grasppose import get_grasp, get_net,vis_grasps
from langrasp.utils import show_image, create_pointcloud_from_rgbd
from langrasp.thirdpart.moge.moge.model.v2 import MoGeModel
from langrasp.depth_optimizer import optimize_depth_map


os.environ["XDG_SESSION_TYPE"] = "x11"
np.set_printoptions(precision=12, suppress=True)
# 获取配置
config = Config()
data_dir = config.data_dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    np.set_printoptions(precision=6, suppress=True)

loaded_color_bgr = cv2.imread(data_dir + "color.png")
loaded_color_rgb = cv2.cvtColor(loaded_color_bgr, cv2.COLOR_BGR2RGB)
loaded_depth_measure = cv2.imread(data_dir + "depth_measure.png", cv2.IMREAD_ANYDEPTH) 
loaded_depth_infer = cv2.imread(data_dir + "depth_infer.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
loaded_depth_optimized = cv2.imread(data_dir + "depth_optimized.png", cv2.IMREAD_ANYDEPTH) 
    
# show_image(loaded_color_rgb)
show_image(loaded_depth_measure)
show_image(loaded_depth_infer)
#show_image(loaded_depth_optimized)

create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = loaded_depth_measure)
create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = loaded_depth_optimized)
loaded_mask = cv2.imread(data_dir + "mask.png")
show_image(loaded_mask)

graspnet = get_net(checkpoint_path = config.graspnet_checkpoint_path)

for i in range(100):
    target_graspgroup, cloud = get_grasp(graspnet, loaded_color_rgb, loaded_depth_optimized, loaded_mask, config.cam_info)