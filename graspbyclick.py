import os
import cv2
import time

import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from langrasp.config import Config
from langrasp.camera import CameraD435
from langrasp.piper_obj import PiperClass
from langrasp.utils import show_image
# loaded_color_bgr = cv2.imread("/home/yama/docker_share/langrasp/data/color.png")
# loaded_color_rgb = cv2.cvtColor(loaded_color_bgr, cv2.COLOR_BGR2RGB)
# show_image(loaded_color_rgb)
# 全局变量用于存储点击的坐标
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"点击坐标: x={x}, y={y}")
        depth_value = param[y, x]  # 注意：OpenCV图像是 (H, W)，所以是 [y, x]
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

    # 初始化
    # cam = CameraD435()
    # cam.enable_stream(rs.stream.color, *config.cam_info["cam_rgb_hw"], rs.format.bgr8, 30)
    # cam.enable_stream(rs.stream.depth, *config.cam_info["cam_depth_hw"], rs.format.z16, 30)
    # cam.start()
    # time.sleep(2)

    piper = PiperClass(can_name="can_piper", enable_curobo=False)
    #piper.set_ctrl_mode2can()

    # 初始位置
    start_pose = np.array([210, 0, 356, 180, 30, 180])
    # start_joint_angle, _, _ = piper.inverse_kinematics(end_pose=start_pose, method="cpu")
    # piper.control_gripper(70)
    # piper.control_joint(start_joint_angle)
    # time.sleep(3)

    # 循环直到找到合适的抓取点或用户退出
    while True:
        # 获取场景图像
        # scence_dict = cam.get_average_depth(format2numpy=True, n=100)
        # img_rgb_array = scence_dict["color"]
        # img_depth_array = scence_dict["depth"]
        
        # # 获取用户点击的 (x, y) 和对应的深度 z
        # print("\n请选择抓取目标点...")
        # point_xy, depth_z = get_click_point_and_depth(img_rgb_array, img_depth_array)
        
        # # 检查用户是否退出
        # if point_xy is None:
        #     print("用户退出选择，程序结束")
        #     break

        # x, y = point_xy
        # z = depth_z
        x = 458
        y = 346
        z = 715
        print(f"选中的点: (x={x}, y={y}, z={z})")

        # 计算坐标
        K = config.cam_info["intrinsic"]
        p_cog4cam = np.linalg.inv(K) @ np.array([x*z, y*z, z])
        t_cog4cam = np.append(p_cog4cam, 1).reshape(-1, 1)
        print(f"相机坐标系下的重心坐标: {t_cog4cam.tolist()}")
        breakpoint()
        # 计算相对于机械臂基座的坐标
        t_cog4base = config.T_cam2base @ t_cog4cam
        print(f"基座坐标系下的重心坐标: {t_cog4base.tolist()}")
        
        # 生成抓取姿态
        near_end_pose = np.array([t_cog4base[0, 0],t_cog4base[1, 0],t_cog4base[2, 0] + 250, 180, 15, 180])
        grasp_end_pose = np.array([t_cog4base[0, 0], t_cog4base[1, 0], t_cog4base[2, 0] + 250 - 110, 180, 15, 180])
        print(near_end_pose)
        print(grasp_end_pose)
        
        
        # 计算逆运动学
        near_joint_angle, near_trans_err, near_rot_err = piper.inverse_kinematics(end_pose=near_end_pose, method="cpu")
        grasp_joint_angle, grasp_trans_err, grasp_rot_err = piper.inverse_kinematics(end_pose=grasp_end_pose, method="cpu")

        total_error = near_trans_err + near_rot_err + grasp_trans_err + grasp_rot_err
        print(f"总误差: {total_error}")

        if total_error < 0.5:
            print("找到合适的抓取点，执行抓取动作...")
            
            # 移动到预抓取位置
            piper.control_joint(near_joint_angle)
            time.sleep(3)
            
            # 移动到抓取位置
            piper.control_joint(grasp_joint_angle)
            time.sleep(2)
            piper.control_gripper(0)  # 闭合夹爪
            time.sleep(3)

            # 抬起到初始位置
            piper.control_joint(start_joint_angle)
            time.sleep(3)
            print("抓取完成")
            break
        else:
            print(f"误差过大 ({total_error} ≥ 0.5)，请重新选择点")
            print(f"各分量误差: 预抓取位置平移={near_trans_err}, 预抓取位置旋转={near_rot_err}")
            print(f"           抓取位置平移={grasp_trans_err}, 抓取位置旋转={grasp_rot_err}")
            continue  # 回到循环开始，重新选择点

    # 关闭外设
    cam.stop()
    piper.disconnect()
    print("程序结束")

if __name__ == "__main__":
    main()
