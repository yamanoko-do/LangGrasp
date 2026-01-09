"""
数据配置类，标定的参数，数据的路径等都放在此处
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from langgrasp.camera import CameraD435

@dataclass
class Config:
    # 使用default_factory来创建字典默认值
    cam_info: Dict = field(default_factory=lambda: {
        "cam_rgb_hw":(1280, 720),
        "cam_depth_hw":(640, 480),
        "intrinsic": np.array([[902.3352359413651, 0.0, 642.2675799117687], [0.0, 901.5064189761518, 371.8472754481835], [0.0, 0.0, 1.0]])
    })
    
    data_dir: str = "data/"
    graspnet_checkpoint_path: str = "./data/weights/graspnet.tar"
    sam_checkpoint_path: str = "./data/weights/SAM/sam2.1_b.pt"
    moge_checkpoint_path: str = "./data/weights/moge-2-vitl-normal/model.pt"
    
    # 手眼标定参数
    R_cam2base: np.ndarray = np.array(
            [[-0.060551945799933546, -0.9734855081594408, 0.22058881944331965], [-0.9888259499642099, 0.028342421640963303, -0.14635555272316778], [0.1362229882847239, -0.22698606243353667, -0.9643239211611931]]
    )
    t_cam2base: np.ndarray = np.array([114.97441510966553, 103.97966302256899, 771.5239243609715])

    #AX=XB的X
    R_board2F: np.ndarray = np.array(
            [[-0.9957502610228033, 0.042080638912612725, 0.0819184808314103], [0.0830554355220814, 0.026017964459381177, 0.9962052299379018], [0.039789600441035085, 0.9987753928457075, -0.02940242069364533]]
    )
    t_board2F: np.ndarray = np.array([84.31314137989958, 2.0046493795987765, 152.37193008091572])

    # 按照定义的piper夹爪坐标系和棋盘格坐标系处理griper到board
    #R_griper2board = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R_griper2board = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    t_griper2board = np.array([t_board2F[0], 0, 0])#标定时我夹在中心

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
        #print(self.T_griper2board)


# 使用示例
if __name__ == "__main__":
    # 创建默认配置
    default_config = Config()
    print(default_config)
    print(default_config.int_param)
