"""
使用curobo进行运动学计算,经测试curobo没有显著减小求解时间(使用solve_single,也许solve_batch能解决这个问题),同样的场景这个实现能找到更多解，并且其确实会找到一些原始实现找不到的逆解,但也存在一些解原始实现能找到而curobo找不到的情况
"""

from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from scipy.spatial.transform import Rotation as R

class CuroboKinematic:
    """
    结合机器人正运动学和逆运动学的类（基于curobo）。
    输入和输出均为基本Python列表类型。
    """

    def __init__(self, urdf_path: str = None, target_link_name: str = "link6"):
        """
        初始化运动学求解器。

        参数:
            urdf_path: 机器人URDF文件的路径。
            target_link_name: 末端连杆名称。
        """
        self.tensor_args = TensorDeviceType()

        # curobo机器人配置
        self.robot_cfg = RobotConfig.from_basic(urdf_path, "base_link", target_link_name , self.tensor_args)

        # curobo IK配置
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.01,
            position_threshold=0.0011,
            num_seeds=100,#每个 IK 问题并行优化的种子数量
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)

        # 获取关节信息
        self.joints_cfg = self.robot_cfg.kinematics.get_joint_limits()

        # 关节数量和限制
        self.num_joints = len(self.joints_cfg.joint_names)
        self.joint_limits = [i for i in zip(*self.joints_cfg.position.tolist())]

        self.joints_zero = [0] * self.num_joints
        self.xyz_wxyz_zero = self.solve_fk(self.joints_zero)
        self.solve_ik(self.xyz_wxyz_zero)

    def solve_fk(self, joint_angles: list) -> list:
        """
        正运动学：输入关节角度列表（度），输出末端位姿（xyz单位为毫米 + 欧拉角单位为度）。

        参数:
            joint_angles: 关节角度列表（单位：度）。

        返回:
            [x, y, z, rx, ry, rz]列表（x,y,z单位为毫米；rx, ry, rz单位为度，欧拉角顺序：xyz）
        """
        # 1. 将输入的角度（度）转换为弧度并转为tensor
        q_deg_tensor = self.tensor_args.to_device(joint_angles)
        q_rad_tensor = torch.deg2rad(q_deg_tensor)

        # 2. 计算正运动学
        kin_state = self.ik_solver.fk(q_rad_tensor)

        # 3. 位置：米 -> 毫米
        xyz_m = kin_state.ee_position[0].tolist()
        xyz_mm = [coord * 1000 for coord in xyz_m]  # 1米 = 1000毫米

        # 4. 姿态：四元数（wxyz）-> 欧拉角（度，xyz顺序）
        quat_wxyz = kin_state.ee_quaternion[0].tolist()  # 原始四元数格式：[w, x, y, z]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  # 转换为scipy要求的[x, y, z, w]格式

        # 四元数转欧拉角（采用xyz顺序）
        rot = R.from_quat(quat_xyzw)  # 初始化旋转对象
        euler_deg = rot.as_euler('xyz', degrees=True).tolist()

        return xyz_mm + euler_deg

    def solve_ik(
        self,
        end_pose: list,
        initial_guess: list = None,
        newton_iters: int = 1000,
    ) -> list:
        """
        逆运动学：输入目标末端位姿[x, y, z, rx, ry, rz]（xyz单位为毫米 + 欧拉角单位为度），
        输出关节角度列表（度）。

        参数:
            end_pose: 目标末端位姿[x, y, z, rx, ry, rz]（位置单位为毫米，欧拉角单位为度，欧拉角顺序：xyz）。
            initial_guess: 初始猜测的关节角度列表（度）。

        返回:
            关节角度列表（度），若未找到解则返回None。
        """
        # 将末端位姿拆分为位置和欧拉角
        target_position = end_pose[:3]  # [x, y, z] 单位为毫米
        target_euler = end_pose[3:]     # [rx, ry, rz] 单位为度
        
        # 将位置从毫米转换为米
        pos_meters = [coord / 1000.0 for coord in target_position]
        
        # 将欧拉角（度）转换为四元数（wxyz）
        euler_rad = [torch.deg2rad(torch.tensor(angle)).item() for angle in target_euler]
        rot = R.from_euler('xyz', euler_rad, degrees=False)
        quat_xyzw = rot.as_quat()  # 返回[x, y, z, w]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # 转换为[w, x, y, z]
        
        # 创建目标位姿
        pos_tensor = self.tensor_args.to_device(pos_meters)
        quat_tensor = self.tensor_args.to_device(quat_wxyz)
        goal = Pose(pos_tensor, quat_tensor)

        # 若提供初始猜测，将其从度转换为弧度
        seed_config = None
        if initial_guess is not None:
            seed_rad = torch.deg2rad(self.tensor_args.to_device(initial_guess))
            seed_config = seed_rad.unsqueeze(0).unsqueeze(0)

        # 调用curobo的IK求解器
        result = self.ik_solver.solve_single(goal, seed_config=seed_config, newton_iters=newton_iters)
        
        if result.success[0]:
            # 将解从弧度转换为度
            solution_rad = result.solution[0][0].tolist()
            solution_deg = [torch.rad2deg(torch.tensor(angle)).item() for angle in solution_rad]
            return solution_deg
        else:
            return None

# 测试示例
if __name__ == "__main__":
    urdf_path = "/root/host_share/curobo/piper.urdf"
    solver = CuroboKinematic(urdf_path=urdf_path)

    # 测试正运动学
    joint_angles = [5.598, 6.135, -27.281, -9.023, 37.24, -92.193]
    fk_solution = solver.solve_fk(joint_angles)
    print(f"正运动学解: {fk_solution}")

    # 测试逆运动学
    ik_solution = solver.solve_ik(fk_solution)
    print(f"逆运动学解: {ik_solution}")

    # Calculate time
    import time
    t1 = time.time()
    while True:
        ik_solution = solver.solve_ik(fk_solution)
        print(f"IK solution: {ik_solution}")
        t2 = time.time()
        print(f"time: {(t2 - t1) * 1000:.2f} ms")
        t1 = t2