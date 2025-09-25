"""
piper机械臂类,是对piper_sdk的包装,通过此类和piper交互
"""
import time
from piper_sdk import *
from typing import List,Tuple
import numpy as np
import math
import cv2


class PiperClass():
    def __init__(self,can_name = "can_piper"):
        self.piper = C_PiperInterface_V2(can_name)
        self.piper.ConnectPort()
        while( not self.piper.EnablePiper()):#使能机械臂
            time.sleep(0.01)
        print("init: 成功连接到piper")
        print(f"init: 当前机械臂状态：{self.piper.GetArmStatus().arm_status.ctrl_mode}")

    def control_gripper(self, length):
        """
        输入mm,范围0-70
        """
        # 限制输入范围在 0 ~ 70 mm
        length = max(0, min(length, 70))
        while( not self.piper.EnablePiper()):
            time.sleep(0.01)
        self.piper.GripperCtrl(0,1000,0x02, 0)
        self.piper.GripperCtrl(0,1000,0x01, 0)
        while( not self.piper.EnablePiper()):
            time.sleep(0.01)
        self.piper.GripperCtrl(0,1000,0x02, 0)
        self.piper.GripperCtrl(0,1000,0x01, 0)
        #转换到0.001mm
        range = length* 1000
        #四舍五入
        range = round(range)
        self.piper.GripperCtrl(abs(range), 1000, 0x01, 0)

    def control_joint(self, joint_angle):
        """
        输入关节角，单位度
        """
        #转换成0.001度并四舍五入
        factor = 1000
        joint_0 = round(joint_angle[0]*factor)
        joint_1 = round(joint_angle[1]*factor)
        joint_2 = round(joint_angle[2]*factor)
        joint_3 = round(joint_angle[3]*factor)
        joint_4 = round(joint_angle[4]*factor)
        joint_5 = round(joint_angle[5]*factor)
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        
    def getpose(self) -> dict:
        """
        获取机械臂末端位姿和关节角度,单位: mm和度
        """
        endpose = self.piper.GetArmEndPoseMsgs().end_pose
        pose_list = [endpose.X_axis, endpose.Y_axis, endpose.Z_axis, endpose.RX_axis, endpose.RY_axis, endpose.RZ_axis]
        pose_list = [number / 1000 for number in pose_list]

        joint_state = self.piper.GetArmJointMsgs().joint_state
        joint_state_list = [joint_state.joint_1, joint_state.joint_2, joint_state.joint_3, joint_state.joint_4, joint_state.joint_5, joint_state.joint_6]
        joint_state_list = [number / 1000 for number in joint_state_list]

        pose_dict = {
            "end_pose": pose_list,
            "joint_state": joint_state_list
        }
        return pose_dict

    def disconnect(self):
        """
        断开机械臂连接
        """
        return self.piper.DisconnectPort()
    def disable(self):
        """
        失能机械臂 !!!注意，这会导致机械臂自由落体!!!
        """
        while(self.piper.DisablePiper()):
            time.sleep(0.01)
        print("失能成功!!!!")
    
    def get_ctrl_mode(self):   
        """
        获取机械臂状态
        """
        return self.piper.GetArmStatus().ctrl_mode
    
    def set_ctrl_mode2can(self):
        """
        切换机械臂到can控制模式
        """
        print(f"当前机械臂状态为{self.piper.GetArmStatus().arm_status.ctrl_mode}")

        if self.piper.GetArmStatus().arm_status.ctrl_mode == 0x00:#如果是待机
            print("尝试从 待机模式->can控制模式")
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
            time.sleep(1)
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
        elif self.piper.GetArmStatus().arm_status.ctrl_mode == 0x02:#如果是示教模式
            print("尝试从 示教模式->can控制模式")
            self.piper.MotionCtrl_1(0x02,0,0)#恢复，示教模式->待机模式(会恢复到别的模式吗？没有我可就写死了)
            # print(piper.GetArmStatus().arm_status.ctrl_mode)
            time.sleep(1)#这里必须要等切换到待机模式
            # print(piper.GetArmStatus().arm_status.ctrl_mode)

            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            time.sleep(1)#这里也必须要等

            while( not self.piper.EnablePiper()):#使能机械臂
                time.sleep(0.01)

            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            while(self.piper.GetArmStatus().arm_status.ctrl_mode != 0x01):#等待进入can控制模式
                time.sleep(0.01)
            print("成功切换到can控制模式")


        elif self.piper.GetArmStatus().arm_status.ctrl_mode == 0x01:#如果是can控制模式
            print("can控制模式") 
            while( not self.piper.EnablePiper()):#使能机械臂
                time.sleep(0.01)
            
        else:
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
        #清除夹爪错误使其可以被正常控制
        self.piper.GripperCtrl(0,1000,0x02, 0)
        self.piper.GripperCtrl(0,1000,0x01, 0)

    @staticmethod
    def get_dh_params():
        dh_params = [
        #(a, alpha, d, theta_offset)
        # (0,         0 ,         123,    0),
        # (0,         -np.pi/2,   0,      (-172.22/180)*np.pi),
        # (285.03,    0,          0,      (-102.78/180)*np.pi),
        # (-21.98,    np.pi/2,    250.75, 0),
        # (0,         -np.pi/2,   0,      0),
        # (0,         np.pi/2,    91,     0)
        (0,         0 ,         123,    0),
        (0,         -np.pi/2,   0,      (-174.22/180)*np.pi),
        (285.03,    0,          0,      (-100.78/180)*np.pi),
        (-21.98,    np.pi/2,    250.75, 0),
        (0,         -np.pi/2,   0,      0),
        (0,         np.pi/2,    91,     0)
        ]
        return dh_params
    
    @staticmethod
    def dh_transform(a, alpha, d, theta):
        """
        根据DH参数,计算现代DH矩阵
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct,    -st,    0,      a],
            [st*ca, ct*ca,  -sa,    -d*sa],
            [st*sa, ct*sa,  ca,     d*ca],
            [0 ,      0,      0,    1]
        ])
    
    @staticmethod
    def forward_kinematics(joint_angles, format = "matrix"):
        """
        参数:
        joint_angles : 长度为6的关节角度列表 单位为度
        format (str, optional): 输出位姿的格式，默认为"matrix"
            - "matrix": 返回4x4齐次变换矩阵
            - "euler": 返回欧拉角形式的位姿mm和度
        返回：计算得到的末端位姿
        """
        dh_params = PiperClass.get_dh_params()
        T = np.eye(4)
        for i in range(6):
            a, alpha, d, theta_offset = dh_params[i]
            theta_rad = math.radians(joint_angles[i]) + theta_offset
            T_i = PiperClass.dh_transform(a, alpha, d, theta_rad)
            T = T @ T_i
        if format == "matrix":
            return T
        elif format == "euler":
            return PiperClass.matrix_to_pose(T,format2deg=True)
    

    @staticmethod
    def matrix_to_pose(T,format2deg = False):
        """
        从齐次变换矩阵中计算[x, y, z, rx, ry, rz],位移mm,角度单位deg和rad可选
        
        """
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        R = T[:3, :3]
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            rx = math.atan2(R[2,1], R[2,2])
            ry = math.atan2(-R[2,0], sy)
            rz = math.atan2(R[1,0], R[0,0])
        else:
            print("万向节锁")
            # 万向节锁：按参考实现处理（设 rx = 0）
            # rx = 0.0
            # ry = math.atan2(-R[2,0], sy)
            # rz = math.atan2(-R[0,1], R[0,2])
        if format2deg:
            rx, ry, rz = np.degrees([rx, ry, rz])
        return [x, y, z, rx, ry, rz]

    @staticmethod
    def inverse_kinematics(end_pose, initial_guess=None, max_iterations=100, tolerance=1e-4):
        """
        使用 scipy.optimize.least_squares 的数值逆运动学
        - end_pose: [x, y, z, rx, ry, rz]   # 位置单位 mm，角度单位度
        - initial_guess(可选): 关节角初始值（度）
        返回: 成功时返回关节角列表（度），失败时返回 None
        """
        dh_params = PiperClass.get_dh_params()

        if initial_guess is None:
            initial_guess = [-0.091, 4.551, -41.699, -1.131, 64.722, -85.087]

        # --- 转换输入为弧度 ---
        q0 = np.radians(initial_guess)
        target_pose = np.array(end_pose, dtype=float)
        target_pose[3:] = np.radians(target_pose[3:])   # 姿态角度转为弧度

        def pose_error_deg(pose1, pose2):
            """
            计算两个6D位姿 [x,y,z,rx,ry,rz] (角度单位:度) 之间的误差
            返回: (平移误差 mm, 旋转误差 度)
            """
            # 平移部分
            t1 = np.array(pose1[:3])
            t2 = np.array(pose2[:3])
            trans_error = np.linalg.norm(t1 - t2)

            # 旋转部分：转为旋转矩阵 -> 相对旋转 -> 轴角 -> 角度大小
            def euler_to_matrix(rx, ry, rz):
                # ZYX 顺序（与 matrix_to_pose 一致）
                Rx = np.array([[1, 0, 0],
                            [0, np.cos(rx), -np.sin(rx)],
                            [0, np.sin(rx), np.cos(rx)]])
                Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                            [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]])
                Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                            [np.sin(rz), np.cos(rz), 0],
                            [0, 0, 1]])
                return Rz @ Ry @ Rx

            R1 = euler_to_matrix(np.radians(pose1[3]), np.radians(pose1[4]), np.radians(pose1[5]))
            R2 = euler_to_matrix(np.radians(pose2[3]), np.radians(pose2[4]), np.radians(pose2[5]))

            # 相对旋转: R_diff = R1 @ R2.T （从 pose2 到 pose1 的旋转）
            R_diff = R1 @ R2.T

            # 使用 Rodrigues 转换为旋转向量
            rvec, _ = cv2.Rodrigues(R_diff.astype(np.float64))
            rot_error_deg = np.linalg.norm(rvec) * 180 / np.pi

            return trans_error, rot_error_deg

        # 定义误差函数 (内部全弧度)
        def pose_error_rad(joint_state_rad, target_pose):
            T = PiperClass.forward_kinematics(np.degrees(joint_state_rad), format="matrix")
            pose_curr = PiperClass.matrix_to_pose(T, format2deg=False)  # 输出 [x,y,z,rx,ry,rz] (rx,ry,rz in rad)

            res = np.array(pose_curr) - target_pose
            # 姿态误差归一化到 [-pi, pi]
            res[3:] = (res[3:] + np.pi) % (2*np.pi) - np.pi
            return res

        # 弧度下的关节约束
        lower_bounds = np.radians([-154, 0, -175, -102, -75, -120])
        upper_bounds = np.radians([154, 195, 0, 102, 75, 120])

        # --- 优化 ---
        try:
            from scipy.optimize import least_squares
            res = least_squares(
                pose_error_rad, q0,
                args=(target_pose,),
                bounds=(lower_bounds, upper_bounds),
                max_nfev=max_iterations * len(q0),
                xtol=tolerance, ftol=tolerance, gtol=tolerance
            )
            if res.success:
                q_sol = np.degrees(res.x)   # 返回角度
                cal_pose = PiperClass.forward_kinematics(q_sol,format="euler")
                # print(f"计算关节角度:{q_sol.tolist()}") 
                # print(f"计算关节位姿:{cal_pose}") 
                # print(f"期望位姿:{end_pose}")
                trans_err, rot_err = pose_error_deg(cal_pose, end_pose)
                # print(f"位移误差: {trans_err:.4f} mm")
                # print(f"旋转误差: {rot_err:.4f} 度")
                #q_sol = np.clip(q_sol, -180.0, 180.0)
                return q_sol.tolist(),trans_err,rot_err
            else:
                return None

        except Exception as e:
            print(e)
            print("falling back to Jacobian method...")

            def compute_jacobian_and_pose(joints_rad):
                T = np.eye(4)
                Tees = []
                for i in range(len(dh_params)):
                    a, alpha, d, theta_offset = dh_params[i]
                    theta = joints_rad[i] + theta_offset
                    T_i = PiperClass.dh_transform(a, alpha, d, theta)
                    T = T @ T_i
                    Tees.append(T.copy())

                z_axes = [T_i[0:3, 2] for T_i in Tees]
                origins = [T_i[0:3, 3] for T_i in Tees]
                final_T = Tees[-1]
                final_pos = final_T[0:3, 3]
                final_rot = PiperClass.matrix_to_pose(final_T, format2deg=False)[3:]  # rad

                J = np.zeros((6, len(dh_params)))
                for i in range(len(dh_params)):
                    Ji_t = np.cross(z_axes[i], final_pos - origins[i])
                    Ji_r = z_axes[i]
                    J[0:3, i] = Ji_t
                    J[3:6, i] = Ji_r
                return J, np.concatenate([final_pos, final_rot])

            q = q0.copy()
            for _ in range(max_iterations):
                J, current = compute_jacobian_and_pose(q)
                pos_error = target_pose[:3] - current[:3]
                rot_error = (target_pose[3:] - current[3:] + np.pi) % (2*np.pi) - np.pi
                error = np.concatenate([pos_error, rot_error])

                if np.linalg.norm(error) < tolerance:
                    return np.clip(np.degrees(q), -180.0, 180.0).tolist()

                try:
                    dq = np.linalg.pinv(J) @ error * 0.1
                except np.linalg.LinAlgError:
                    return None

                q += dq
                q = np.clip(q, lower_bounds, upper_bounds)

            return None

if __name__ == "__main__":
    """
    获取机械臂位姿
    """
    piper = PiperClass(can_name = "can_piper")
    piper.set_ctrl_mode2can()
    try:
        while True:
            pose_list = piper.getpose()
            print(pose_list)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序已被用户中断，正在退出...")
    finally:
        piper.disable()
        # 这里可以添加一些资源释放或清理的代码
        print("程序已退出")


    # """
    # 前向和逆向误差分析
    # """
    # import re
    # import os
    # def read_poses_from_txt(file_path):
    #     """
    #     读取txt文件中的一行多个字典格式的数据
    #     返回: 列表，每个元素是一个包含 end_pose 和 joint_state 的字典
    #     """
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         content = f.read()

    #     # 使用正则提取所有字典
    #     pattern = r"\{'end_pose':\s*\[([^\]]+)\],\s*'joint_state':\s*\[([^\]]+)\]\}"
    #     matches = re.findall(pattern, content)

    #     data_list = []
    #     for match in matches:
    #         end_pose = [float(x.strip()) for x in match[0].split(',')]
    #         joint_state = [float(x.strip()) for x in match[1].split(',')]
    #         data_list.append({
    #             'end_pose': end_pose,
    #             'joint_state': joint_state
    #         })
    #     return data_list
    # # 读取数据文件
    # file_path = "./data/eye2hand_images/pose_data.txt"  # 修改为你的实际路径
    
    # if not os.path.exists(file_path):
    #     print(f"文件 {file_path} 不存在！")


    # data_list = read_poses_from_txt(file_path)
    # print(f"共加载 {len(data_list)} 组数据\n")

    # trans_errors = []
    # angle_errors = []

    # for idx, data in enumerate(data_list):
    #     print(f"--- 数据 {idx+1} ---")

    #     true_end_pose = data['end_pose']     # [x, y, z, rx, ry, rz] (mm, deg)
    #     true_joint_state = data['joint_state']  # (deg)

    #     # ========================
    #     # 1. 正向运动学验证
    #     # ========================
    #     fk_calculated = PiperClass.forward_kinematics(true_joint_state, format = "euler")
    #     print(f"末端真实值{true_end_pose}")
    #     print(f"末端计算值{fk_calculated}")
    #     trans_err = np.linalg.norm(np.array(true_end_pose[:3]) - np.array(fk_calculated[:3]))
    #     angle_err = np.linalg.norm(np.array(true_end_pose[3:]) - np.array(fk_calculated[3:]))

    #     trans_errors.append(trans_err)
    #     angle_errors.append(angle_err)

    #     print(f"FK 平移误差: {trans_err:.4f} mm")
    #     print(f"FK 角度误差: {angle_err:.4f} °")

    #     # ========================
    #     # 2. 逆向运动学验证
    #     # ========================
    #     ik_result = PiperClass.inverse_kinematics(
    #         end_pose=true_end_pose,
    #         #initial_guess=true_joint_state,
    #         max_iterations=1000
    #     )
        
    #     if ik_result is None:
    #         print("IK 求解失败")
    #         continue
    #     print(f"IK解关节角度: {ik_result} °")
    #     print(f"真实关节角度: {true_joint_state}°")

    #     # 用 IK 结果再做一次 FK 验证
    #     fk_from_ik = PiperClass.forward_kinematics(ik_result, format = "euler")
    #     ik_trans_err = np.linalg.norm(np.array(true_end_pose[:3]) - np.array(fk_from_ik[:3]))
    #     ik_angle_err = np.linalg.norm(np.array(true_end_pose[3:]) - np.array(fk_from_ik[3:]))

    #     print(f"IK 后 FK 验证 - 平移误差: {ik_trans_err:.4f} mm")
    #     print(f"IK 后 FK 验证 - 角度误差: {ik_angle_err:.4f} °")

    #     joint_diff = np.max(np.abs(np.array(ik_result) - np.array(true_joint_state)))
    #     print(f"最大关节角偏差: {joint_diff:.4f} °")
    #     print()

    # # 统计汇总
    # print("========== 总体误差统计 ==========")
    # if trans_errors:
    #     print(f"平均 FK 平移误差: {np.mean(trans_errors):.4f} mm")
    #     print(f"平均 FK 角度误差: {np.mean(angle_errors):.4f} °")
    #     print(f"最大 FK 平移误差: {np.max(trans_errors):.4f} mm")
    #     print(f"最大 FK 角度误差: {np.max(angle_errors):.4f} °")
    