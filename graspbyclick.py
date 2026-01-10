import os
import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from langgrasp.config import Config
from langgrasp.camera import CameraD435
from langgrasp.piper_obj import PiperClass
from langgrasp.utils import show_image, create_pointcloud_from_rgbd
from langgrasp.thirdpart.moge.moge.model.v2 import MoGeModel
from langgrasp.depth_optimizer import optimize_depth_map

# 全局变量用于存储点击的坐标
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"点击坐标: x={x}, y={y}")
        depth_img, rgb_img = param
        depth_value = depth_img[y, x]  # 注意：OpenCV图像是 (H, W)，所以是 [y, x]
        print(f"对应深度值 z = {depth_value}")
        # 可选：在图像上标记点击点
        cv2.circle(param[1], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("RGB Image", param[1])

def get_click_point_and_depth(img_rgb, img_depth):
    global clicked_point
    clicked_point = None

    # 创建窗口并设置鼠标回调
    cv2.namedWindow("RGB Image")
    # 将 depth 图像作为额外参数传入（用于读取深度），同时传入 rgb 图像用于显示
    cv2.setMouseCallback("RGB Image", mouse_callback, (img_depth, img_rgb.copy()))

    # 显示图像
    cv2.imshow("RGB Image", img_rgb)
    print("请在图像上点击一个点，按任意键退出...")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if clicked_point is not None or key != 255:  # 按任意键也可退出
            break

    cv2.destroyAllWindows()
    if clicked_point:
        return clicked_point, img_depth[clicked_point[1], clicked_point[0]]
    else:
        return None, None  # 用户按任意键未点击点退出

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

def main():
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    np.set_printoptions(precision=12, suppress=True)

    # 获取配置
    config = Config()
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        np.set_printoptions(precision=6, suppress=True)


    # 初始化机械臂
    cam = CameraD435()
    cam.enable_stream(rs.stream.color, *config.cam_info["cam_rgb_hw"], rs.format.bgr8, 30)
    cam.enable_stream(rs.stream.depth, *config.cam_info["cam_depth_hw"], rs.format.z16, 30)
    cam.start()
    time.sleep(2)
    piper = PiperClass(can_name="can_piper", enable_curobo=False)
    piper.set_ctrl_mode2can()

    # 初始化模型
    device = torch.device("cuda")
    moge_model = MoGeModel.from_pretrained(config.moge_checkpoint_path).to(device) 

    # 初始位置
    start_pose = np.array([210, 0, 356, 180, 30, 180])
    start_joint_angle, _, _ = piper.inverse_kinematics(end_pose=start_pose, method="cpu")
    piper.control_gripper(70)
    piper.control_joint(start_joint_angle)
    time.sleep(3)

    # 循环直到找到合适的抓取点或用户退出
    while True:
        # 获取场景图像
        
        scence_dict = cam.get_kalman_depth(format2numpy=True, n=100)
        img_rgb_array = scence_dict["color"]
        img_depth_array = scence_dict["depth"]
        #show_image(img_depth_array)
        img_depth_array[img_depth_array > 10000] = 0
        cv2.imwrite(data_dir+"color.png", img_rgb_array)
        cv2.imwrite(data_dir+"depth_measure.png", img_depth_array)

        #load
        loaded_color_bgr = cv2.imread(data_dir + "color.png")
        loaded_color_rgb = cv2.cvtColor(loaded_color_bgr, cv2.COLOR_BGR2RGB)
        loaded_depth_measure = cv2.imread(data_dir + "depth_measure.png", cv2.IMREAD_ANYDEPTH) 
        #show_image(loaded_depth_measure)
        #create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = loaded_depth_measure)
        #使用moge对深度图修正
        moge_rgb_input = torch.tensor(loaded_color_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)  
        moge_st = time.time() 
        moge_output = moge_model.infer(moge_rgb_input)
        moge_et = time.time()
        print(f"单目深度估计耗时: {moge_et-moge_st:.4f} 秒")
        depth_infer = (moge_output["depth"] * 1000).cpu().to(torch.uint16).numpy()
        cv2.imwrite(data_dir+"depth_infer.png", depth_infer)
        loaded_depth_infer = cv2.imread(data_dir + "depth_infer.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_optimized_st = time.time() 
        depth_optimized = optimize_depth_map(loaded_depth_measure, loaded_depth_infer, loaded_color_rgb, config.cam_info["intrinsic"])
        depth_optimized_et = time.time()
        print(f"深度优化耗时: {depth_optimized_et-depth_optimized_st:.4f} 秒")
        cv2.imwrite(data_dir+"depth_optimized.png", depth_optimized)
        loaded_depth_optimized = cv2.imread(data_dir + "depth_optimized.png", cv2.IMREAD_ANYDEPTH) 
        #show_image(loaded_depth_optimized)

        #可视化修正的场景点云
        
        #create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = depth_infer)
        
        
        # 获取用户点击的 (x, y) 和对应的深度 z
        print("\n请选择抓取目标点...")
        point_xy, depth_z = get_click_point_and_depth(img_rgb_array, loaded_depth_optimized)
        
        # 检查用户是否退出
        if point_xy is None:
            print("用户退出选择，程序结束")
            break

        x, y = point_xy
        z = depth_z
        # x = 458
        # y = 346
        # z = 715
        print(f"选中的点: (x={x}, y={y}, z={z})")

        # 计算相机坐标系下的重心位置
        K = config.cam_info["intrinsic"]
        p_cog2cam = np.linalg.inv(K) @ np.array([x*z, y*z, z])
        t_cog2cam = np.append(p_cog2cam, 1).reshape(-1, 1)
        print(f"相机坐标系下的重心位置: {t_cog2cam.tolist()}")
        #计算基座坐标系下的抓取点位置
        t_grasppose2base = config.T_cam2base @ t_cog2cam
        print(t_grasppose2base)

        #计算基座坐标系下的抓取点姿态
        rx, ry, rz = np.deg2rad([180, 15, 180])
        R_grasppose2base = euler_to_matrix(rx, ry, rz)
        T_grasppose2base = np.eye(4)
        T_grasppose2base[:3, :3] = R_grasppose2base
        T_grasppose2base[:3, 3] = t_grasppose2base[:3, 0]

        leftright_offset = -20  # 夹爪开合方向的偏移修正系数,大于0夹爪向（标定板z负/向夹爪y负）方向移动
        depth_offset = -55    # 越小抓的约深，如-20比-10更深,大于0夹爪向（标定板y负/向夹爪z负）方向移动
        updown_offset = 0    # 垂直夹爪开合方向的偏移修正系数,大于0夹爪向（标定板x负/向夹爪x负）方向移动
        #计算基座下法兰盘的接近位姿
        T_gripernear2board = config.T_griper2board.copy()
        T_gripernear2board[1, 3] = depth_offset+50
        T_gripernear2board[2, 3] = leftright_offset

        T_gripernear2F = config.T_board2F @ T_gripernear2board
        T_Fnear2base = T_grasppose2base @ np.linalg.inv(T_gripernear2F)

        #计算基座下法兰盘的抓取位姿
        T_gripertarget2board = config.T_griper2board.copy()
        T_gripertarget2board[1, 3] = depth_offset
        T_gripertarget2board[2, 3] = leftright_offset

        T_gripertarget2F = config.T_board2F @ T_gripertarget2board
        T_Ftarget2base = T_grasppose2base @ np.linalg.inv(T_gripertarget2F)

        # 将位姿转换成欧拉角表示
        Euler_Fnear2base = PiperClass.matrix_to_pose(T_Fnear2base, format2deg=True)
        Euler_Ftarget2base = PiperClass.matrix_to_pose(T_Ftarget2base, format2deg=True)
        

        # 计算逆运动学
        near_joint_angle, near_trans_err, near_rot_err = piper.inverse_kinematics(end_pose=Euler_Fnear2base, method="cpu")
        grasp_joint_angle, grasp_trans_err, grasp_rot_err = piper.inverse_kinematics(end_pose=Euler_Ftarget2base, method="cpu")


        total_error = near_trans_err + near_rot_err + grasp_trans_err + grasp_rot_err
        print(f"总误差: {total_error}")
        trans_ok = near_trans_err + grasp_trans_err < 0.3
        rot_ok   = near_rot_err + grasp_rot_err < 35        
        #允许角度的逆解轻微误差，但位置要准
        if trans_ok and rot_ok:
            print("找到合适的抓取点，执行抓取动作...")
            
            # 移动到预抓取位置
            piper.control_joint(near_joint_angle)
            time.sleep(3)
            
            # 移动到抓取位置
            piper.control_joint(grasp_joint_angle)
            time.sleep(2)
            piper.control_gripper(length=0, effort=0.5)  # 闭合夹爪
            time.sleep(3)

            # 抬起到初始位置
            piper.control_joint(start_joint_angle)
            time.sleep(3)
            print("抓取完成")
            break
        else:
            print(f"误差过大 ({total_error} )，请重新选择点")
            print(f"各分量误差: 预抓取位置平移={near_trans_err}, 预抓取位置旋转={near_rot_err}")
            print(f"           抓取位置平移={grasp_trans_err}, 抓取位置旋转={grasp_rot_err}")
            continue  # 回到循环开始，重新选择点

    # 关闭外设
    cam.stop()
    piper.disconnect()
    print("程序结束")

if __name__ == "__main__":
    main()
