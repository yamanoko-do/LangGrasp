"""
数据配置类，标定的参数，数据的路径等都放在此处
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from langrasp.camera import CameraD435

@dataclass
class Config:
    # 使用default_factory来创建字典默认值
    cam_info: Dict = field(default_factory=lambda: {
        "cam_rgb_hw":(1280, 720),
        "cam_depth_hw":(640, 480),
        "intrinsic": np.array([[896.4105731015486, 0.0, 646.5148650122112], [0.0, 895.8013693817877, 374.5274283245406], [0.0, 0.0, 1.0]])
    })
    
    data_dir: str = "data/"
    graspnet_checkpoint_path: str = "./data/weights/graspnet.tar"
    sam_checkpoint_path: str = "./data/weights/SAM/sam2.1_b.pt"
    moge_checkpoint_path: str = "./data/weights/moge-2-vitl-normal/model.pt"
    
    # 手眼标定参数
    R_cam2base: np.ndarray = np.array(
           [[-0.07119741223634257, -0.9822316537498458, 0.1736430443832734], [-0.9926321410203794, 0.052659611468735434, -0.10912560622172676], [0.09804264941414956, -0.18013312769117454, -0.9787439375056426]]
    )
    t_cam2base: np.ndarray = np.array([136.80008629263062, 90.81363485325187, 767.3111892075818])

    #AX=XB的X
    R_board2F: np.ndarray = np.array(
           [[-0.9898673305788128, 0.01613790251665216, -0.14107528470692052], [-0.1408763884630423, 0.01286648997370251, 0.9899435825386046], [0.017790756768212672, 0.9997869880820214, -0.010462668660273459]]
    )
    t_board2F: np.ndarray = np.array( [76.76648791301952, 13.228801083436444, 159.99379878490652])

    # 按照定义的piper夹爪坐标系处理griper到board
    R_griper2board = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    t_griper2board = np.array( [t_board2F[0], 0, 0])#标定时我夹在中心

    def __post_init__(self):
        # #在初始化后处理需要计算的参数
        # intrinsic,_,_,_ = CameraD435.get_intrinsics(
        #     *self.cam_info["cam_rgb_hw"],
        #     *self.cam_info["cam_depth_hw"]
        # )
        # self.cam_info["intrinsic"] = intrinsic

        self.T_cam2base = np.eye(4)
        self.T_cam2base[:3, :3] = self.R_cam2base
        self.T_cam2base[:3, 3] = self.t_cam2base

        self.T_board2F = np.eye(4)
        self.T_board2F[:3, :3] = self.R_board2F
        self.T_board2F[:3, 3] = self.t_board2F

        self.T_griper2board = np.eye(4)
        self.T_griper2board[:3, :3] = self.R_griper2board
        self.T_griper2board[:3, 3] = self.t_griper2board


# 使用示例
if __name__ == "__main__":
    # 创建默认配置
    default_config = Config()
    print(default_config)
    print(default_config.int_param)
