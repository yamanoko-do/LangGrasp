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
            [[0.1272033110949011, -0.978941311300194, 0.1596659847191973], [-0.9701493481431956, -0.0892860257169147, 0.2254733862543689], [-0.20646927118539227, -0.18358081229130596, -0.9610767531340244]]
    )
    t_cam2base: np.ndarray = np.array([111.26420723234966, -118.66936046328698, 894.3611073853308])

    #AX=XB的X
    R_board2F: np.ndarray = np.array(
            [[-0.9985239930407424, -0.033789267551886576, 0.042522002778320175], [0.03963317046866507, 0.0819970091579061, 0.995844215873025], [-0.03713552370135868, 0.9960596246648414, -0.08053680520085559]]
    )
    t_board2F: np.ndarray = np.array([61.91873123862654, 19.265743541074194, 182.35373587337784])

    # 按照定义的piper夹爪坐标系和棋盘格坐标系处理griper到board，考虑graspneet坐标变换版本
    R_griper2board = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    #按照定义的piper夹爪坐标系和棋盘格坐标系处理griper到board
    #R_griper2board = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
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
