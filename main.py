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


def find_viable_grasps(graspnet, loaded_color_rgb, loaded_depth, loaded_mask, config, max_retries=5):
    """
    获取并筛选可行的抓取位姿，返回位姿组及其对应的关节角度
    1. 使用graspnet和mask获取一组 目标物体的抓取位姿
    2. 使用逆运动学计算生成的位姿机械臂是否可达，如果都不可达返回第一步直到达到最大重试次数
    
    参数:
        graspnet: graspnet模型
        loaded_color_rgb: 彩色图像
        loaded_depth: 深度图像
        loaded_mask: 掩码
        config: 配置信息
        max_retries: 最大重试次数
    
    返回:
        tuple: (可行的抓取位姿组, 对应的关节角度列表, 点云)，如果未找到则返回(None, None, None)
    """
    for retry_count in range(max_retries):
        # 获取抓取位姿
        target_graspgroup, cloud = get_grasp(graspnet, loaded_color_rgb, loaded_depth, loaded_mask, config.cam_info)
        keep_ids = []
        near_joint_angles = []  
        grasp_joint_angles = []
        
        for i, grasp in enumerate(target_graspgroup):
            # 计算抓取位姿到相机坐标系的变换矩阵
            T_grasppose2cam = np.eye(4)
            T_grasppose2cam[:3, :3] = grasp.rotation_matrix
            T_grasppose2cam[:3, 3] = grasp.translation * 1000
            
            # 将抓取位姿变换到机械臂基坐标系下
            T_grasppose2base = config.T_cam2base @ T_grasppose2cam
            #print(config.T_griper2board)

            

            # 计算法兰盘的接近位姿
            T_griper2F_near = config.T_board2F @ config.T_griper2board
            T_F2base = T_grasppose2base @ np.linalg.inv(T_griper2F_near)
            
            # 计算接近位姿
            Euler_F2base = PiperClass.matrix_to_pose(T_F2base, format2deg=True)
            near_joint_angle, near_trans_err, near_rot_err = PiperClass.inverse_kinematics(Euler_F2base)


            # 计算法兰盘的抓取位姿
            T_griper2board = config.T_griper2board.copy()
            T_griper2board[1, 3] = -grasp.depth*1000*1.39#系数越大抓的越深
            T_griper2F_near = config.T_board2F @ T_griper2board
            T_F2base = T_grasppose2base @ np.linalg.inv(T_griper2F_near)
            #print(T_griper2board)

            # 计算抓取位姿
            Euler_F2base = PiperClass.matrix_to_pose(T_F2base, format2deg=True)
            grasp_joint_angle, grasp_trans_err, grasp_rot_err = PiperClass.inverse_kinematics(Euler_F2base)
  
            # 筛选可达位姿
            if near_trans_err + near_rot_err +grasp_trans_err + grasp_rot_err< 0.5:
                print(f"找到可达接近位姿: {i}")
                print(f"trans_err: {near_trans_err},rot_err: {near_rot_err}")
                print(f"找到可达抓取位姿: {i}")
                print(f"trans_err: {grasp_trans_err},rot_err: {grasp_rot_err}")

                keep_ids.append(i)
                near_joint_angles.append(near_joint_angle)  
                grasp_joint_angles.append(grasp_joint_angle)
        
        # 检查是否有可用位姿
        if len(keep_ids) > 0:
            print(f"存在{len(keep_ids)}个机械臂可达位姿")
            viable_graspgroup = target_graspgroup[keep_ids]
            sort_viable_graspgroup = viable_graspgroup.sort_by_score()
            
            # 按照抓取分数排序关节角度（与位姿排序保持一致）
            # 获取排序后的索引
            sorted_indices = [idx for idx, _ in sorted(enumerate(viable_graspgroup.scores), key=lambda x: x[1], reverse=True)]
            sorted_near_joint_angles = [near_joint_angles[i] for i in sorted_indices]
            sorted_grasp_joint_angles = [grasp_joint_angles[i] for i in sorted_indices]
            
            vis_grasps(gg=sort_viable_graspgroup, cloud=cloud, window_name="viable_graspgroup")
            return sort_viable_graspgroup, sorted_near_joint_angles, sorted_grasp_joint_angles, cloud
        else:
            print(f"抓取位姿生成第 {retry_count}次尝试没有找到可行解，进行第{retry_count+1}/{max_retries}次尝试")
    
    # 所有重试都失败的情况
    print(f"已达到最大重试次数（{max_retries}次），仍未找到可达位姿")
    return None, None, None
    


def main():
    os.environ["XDG_SESSION_TYPE"] = "x11"
    np.set_printoptions(precision=12, suppress=True)
    # 获取配置
    config = Config()
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        np.set_printoptions(precision=6, suppress=True)

    #初始化相机
    cam=CameraD435()
    cam.enable_stream(rs.stream.color, *config.cam_info["cam_rgb_hw"], rs.format.bgr8, 30)
    cam.enable_stream(rs.stream.depth, *config.cam_info["cam_depth_hw"], rs.format.z16, 30)
    cam.start()
    time.sleep(2)

    # 初始化piper
    piper = PiperClass(can_name = "can_piper")
    piper.set_ctrl_mode2can()
    piper.control_gripper(0)
    time.sleep(2)
    piper.control_gripper(70)
    time.sleep(2)
    piper.control_gripper(0)
    time.sleep(2)

    # 初始化网络
    graspnet = get_net(checkpoint_path = config.graspnet_checkpoint_path)
    sammodel = SAM(config.sam_checkpoint_path)
    device = torch.device("cuda")
    moge_model = MoGeModel.from_pretrained(config.moge_checkpoint_path).to(device) 

    # 输入
    user_input = " the Screwdriver"
    scence_dict = cam.get_average_depth(format2numpy=True, n = 1)
    img_rgb_array = scence_dict["color"]
    img_depth_array = scence_dict["depth"]
    #save
    cv2.imwrite(data_dir+"color.png", img_rgb_array)
    cv2.imwrite(data_dir+"depth_measure.png", img_depth_array)

    #load
    loaded_color_bgr = cv2.imread(data_dir + "color.png")
    loaded_color_rgb = cv2.cvtColor(loaded_color_bgr, cv2.COLOR_BGR2RGB)
    loaded_depth_measure = cv2.imread(data_dir + "depth_measure.png", cv2.IMREAD_ANYDEPTH) 
    show_image(loaded_color_rgb)
    show_image(loaded_depth_measure)

    #使用moge对深度图修正
    moge_rgb_input = torch.tensor(loaded_color_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)  
    moge_output = moge_model.infer(moge_rgb_input)
    depth_infer = (moge_output["depth"] * 1000).cpu().to(torch.uint16).numpy()
    cv2.imwrite(data_dir+"depth_infer.png", depth_infer)
    loaded_depth_infer = cv2.imread(data_dir + "depth_infer.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_optimized = optimize_depth_map(loaded_depth_measure, loaded_depth_infer, loaded_color_rgb, config.cam_info["intrinsic"])

    cv2.imwrite(data_dir+"depth_optimized.png", depth_optimized)
    loaded_depth_optimized = cv2.imread(data_dir + "depth_optimized.png", cv2.IMREAD_ANYDEPTH) 
    show_image(loaded_depth_optimized)

    #可视化修正的场景点云
    #create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = depth_infer)
    
    #获取分割
    targer_mask = get_target_mask(loaded_color_rgb,user_input,sammodel)
    save_mask_as_image(targer_mask,data_dir+"mask.png")
    loaded_mask = cv2.imread(data_dir + "mask.png")
    show_image(loaded_mask)
    #生成可行抓取姿态
    viable_graspgroup, near_joint_angle_list, grasp_joint_angle_list, cloud = find_viable_grasps(
        graspnet, 
        loaded_color_rgb, 
        loaded_depth_optimized, 
        loaded_mask, 
        config,
        max_retries=50
    )

    #从可行抓取姿态中找评分高的
    if viable_graspgroup is not None:
        # 初始化最高分数和对应的索引
        max_score = -float('inf')
        best_index = 0
        
        # 遍历所有抓取姿态，找到分数最高的
        for i, grasp in enumerate(viable_graspgroup):
            if grasp.score > max_score:
                max_score = grasp.score
                best_index = i
        
        print(f"最高分数的抓取姿态索引为: {best_index}")
        print(f"对应的分数为: {max_score}")
        
        best_grasp = viable_graspgroup[best_index:best_index+1]
        best_near_joint_angle = near_joint_angle_list[best_index]
        best_grasp_joint_angle = grasp_joint_angle_list[best_index]

    vis_grasps(gg=best_grasp, cloud=cloud, view_num=1, window_name="bestgrasp")


    print("移动到接近点")
    piper.control_gripper(70)
    piper.control_joint(best_near_joint_angle)
    time.sleep(3)
    print("移动到抓取点")
    piper.control_joint(best_grasp_joint_angle)
    time.sleep(2)
    # 闭合夹爪
    piper.control_gripper(0)
    time.sleep(2)

    # 闭合夹爪
    piper.control_joint([-0.091, 4.551, -41.699, -1.131, 64.722, -85.087])
    #time.sleep(2)





    #关闭外设
    # cam.stop()

if __name__ == "__main__":
    main()