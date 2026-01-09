import os
import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from ultralytics import FastSAM, SAM
from langgrasp.config import Config
from langgrasp.camera import CameraD435
from langgrasp.piper_obj import PiperClass
from langgrasp.mask import get_target_mask,save_mask_as_image
from langgrasp.grasppose import get_grasp, get_net,vis_grasps
from langgrasp.utils import show_image, create_pointcloud_from_rgbd
from langgrasp.thirdpart.moge.moge.model.v2 import MoGeModel
from langgrasp.depth_optimizer import optimize_depth_map
from tqdm import tqdm

def find_viable_grasps(piper, graspnet, loaded_color_rgb, loaded_depth, loaded_mask, config, max_retries=5):
    """
    获取并筛选可行的抓取位姿，返回位姿组及其对应的关节角度
    1. 使用graspnet和mask获取一组目标物体的抓取位姿
    2. 分别使用curobo和cpu两种方法计算逆运动学，选择可达位姿更多的方法
    3. 如果都不可达返回第一步直到达到最大重试次数
    
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
        grasp_st = time.time() 
        target_graspgroup, cloud = get_grasp(graspnet, loaded_color_rgb, loaded_depth, config.cam_info,loaded_mask)
        grasp_et = time.time() 
        print(f"抓取推理优化耗时: {grasp_et-grasp_st:.4f} 秒")
        
        # 分别用两种方法计算可达位姿
        methods = ["curobo", "cpu"]
        method_results = {}
        
        for method in methods:
            keep_ids = []
            near_joint_angles = []  
            grasp_joint_angles = []
            
            for i, grasp in tqdm(enumerate(target_graspgroup)):
                # 计算抓取位姿到相机坐标系的变换矩阵
                T_grasppose2cam = np.eye(4)
                T_grasppose2cam[:3, :3] = grasp.rotation_matrix
                T_grasppose2cam[:3, 3] = grasp.translation * 1000
                
                # 将抓取位姿变换到机械臂基坐标系下
                T_grasppose2base = config.T_cam2base @ T_grasppose2cam

                # 计算法兰盘的接近位姿
                T_griper2F_near = config.T_board2F @ config.T_griper2board
                T_F2base = T_grasppose2base @ np.linalg.inv(T_griper2F_near)
                
                # 计算接近位姿
                Euler_F2base = PiperClass.matrix_to_pose(T_F2base, format2deg=True)
                near_joint_angle, near_trans_err, near_rot_err = piper.inverse_kinematics(end_pose=Euler_F2base, method=method)
                
                # 计算法兰盘的抓取位姿
                T_griper2board = config.T_griper2board.copy()
                T_griper2board[1, 3] = -grasp.depth * 1000 * 1.91  # 系数越大抓的越深
                T_griper2F_near = config.T_board2F @ T_griper2board
                T_F2base = T_grasppose2base @ np.linalg.inv(T_griper2F_near)

                # 计算抓取位姿
                Euler_F2base = PiperClass.matrix_to_pose(T_F2base, format2deg=True)
                ik_st = time.time()
                grasp_joint_angle, grasp_trans_err, grasp_rot_err = piper.inverse_kinematics(end_pose=Euler_F2base, method=method)
                ik_et = time.time()
                #print(f"ik耗时 ({method}): {ik_et-ik_st:.4f} 秒")
                
                # 如果有解
                if near_joint_angle and grasp_joint_angle:
                    # 筛选可达位姿
                    if near_trans_err + near_rot_err + grasp_trans_err + grasp_rot_err < 0.5:
                        # print(f"找到可达接近位姿 ({method}): {i}")
                        # print(f"trans_err: {near_trans_err}, rot_err: {near_rot_err}")
                        # print(f"找到可达抓取位姿 ({method}): {i}")
                        # print(f"trans_err: {grasp_trans_err}, rot_err: {grasp_rot_err}")

                        keep_ids.append(i)
                        near_joint_angles.append(near_joint_angle)  
                        grasp_joint_angles.append(grasp_joint_angle)
            
            # 存储每种方法的结果
            method_results[method] = {
                'keep_ids': keep_ids,
                'near_joint_angles': near_joint_angles,
                'grasp_joint_angles': grasp_joint_angles,
                'viable_count': len(keep_ids)
            }
            print(f"方法 {method} 找到 {len(keep_ids)} 个可达位姿")
        
        # 选择可达位姿更多的方法
        best_method = max(methods, key=lambda m: method_results[m]['viable_count'])
        best_result = method_results[best_method]
        
        print(f"选择方法: {best_method}，可达位姿数: {best_result['viable_count']}")
        
        # 检查是否有可用位姿
        if best_result['viable_count'] > 0:
            keep_ids = best_result['keep_ids']
            near_joint_angles = best_result['near_joint_angles']
            grasp_joint_angles = best_result['grasp_joint_angles']
            
            print(f"存在 {len(keep_ids)} 个机械臂可达位姿（使用 {best_method} 方法）")
            viable_graspgroup = target_graspgroup[keep_ids]
            sort_viable_graspgroup = viable_graspgroup.sort_by_score()
            
            # 按照抓取分数排序关节角度（与位姿排序保持一致）
            sorted_indices = [idx for idx, _ in sorted(enumerate(viable_graspgroup.scores), key=lambda x: x[1], reverse=True)]
            sorted_near_joint_angles = [near_joint_angles[i] for i in sorted_indices]
            sorted_grasp_joint_angles = [grasp_joint_angles[i] for i in sorted_indices]
            
            vis_grasps(gg=target_graspgroup[method_results["curobo"]['keep_ids']], cloud=cloud, window_name="viable_graspgroup_curobo")
            vis_grasps(gg=target_graspgroup[method_results["cpu"]['keep_ids']], cloud=cloud, window_name="viable_graspgroup_cpu")
            return sort_viable_graspgroup, sorted_near_joint_angles, sorted_grasp_joint_angles, cloud
        else:
            print(f"抓取位姿生成第 {retry_count} 次尝试没有找到可行解，进行第 {retry_count+1}/{max_retries} 次尝试")
    
    # 所有重试都失败的情况
    print(f"已达到最大重试次数（{max_retries}次），仍未找到可达位姿")
    return None, None, None, None
    


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

    #初始化相机
    # cam=CameraD435()
    # cam.enable_stream(rs.stream.color, *config.cam_info["cam_rgb_hw"], rs.format.bgr8, 30)
    # cam.enable_stream(rs.stream.depth, *config.cam_info["cam_depth_hw"], rs.format.z16, 30)
    # cam.start()
    # time.sleep(2)

    # # 初始化piper
    piper = PiperClass(can_name = "can_piper", enable_curobo=True)
    # piper.set_ctrl_mode2can()
    # print("移动预抓取点")
    
    # piper.control_gripper(0)
    # time.sleep(2)
    # piper.control_gripper(70)
    # time.sleep(2)
    # piper.control_gripper(0)
    # time.sleep(2)

    # 初始化网络
    graspnet = get_net(checkpoint_path = config.graspnet_checkpoint_path)
    #sammodel = SAM(config.sam_checkpoint_path)
    #device = torch.device("cuda")
    #moge_model = MoGeModel.from_pretrained(config.moge_checkpoint_path).to(device) 

    # 输入
    #user_input = "Pliers"
    #scence_dict = cam.get_average_depth(format2numpy=True, n = 100)
    #img_rgb_array = scence_dict["color"]
    #img_depth_array = scence_dict["depth"]
    #save
    #cv2.imwrite(data_dir+"color.png", img_rgb_array)
    #cv2.imwrite(data_dir+"depth_measure.png", img_depth_array)

    #load
    loaded_color_bgr = cv2.imread(data_dir + "color.png")
    loaded_color_rgb = cv2.cvtColor(loaded_color_bgr, cv2.COLOR_BGR2RGB)
    #loaded_depth_measure = cv2.imread(data_dir + "depth_measure.png", cv2.IMREAD_ANYDEPTH) 
    #show_image(loaded_color_rgb)
    #show_image(loaded_depth_measure)
    #create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = loaded_depth_measure)

    #使用moge对深度图修正
    # moge_rgb_input = torch.tensor(loaded_color_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)  
    # moge_st = time.time() 
    # moge_output = moge_model.infer(moge_rgb_input)
    # moge_et = time.time()
    # print(f"单目深度估计耗时: {moge_et-moge_st:.4f} 秒")
    # depth_infer = (moge_output["depth"] * 1000).cpu().to(torch.uint16).numpy()
    # cv2.imwrite(data_dir+"depth_infer.png", depth_infer)
    # loaded_depth_infer = cv2.imread(data_dir + "depth_infer.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    # depth_optimized_st = time.time() 
    # depth_optimized = optimize_depth_map(loaded_depth_measure, loaded_depth_infer, loaded_color_rgb, config.cam_info["intrinsic"])
    # depth_optimized_et = time.time()
    # print(f"深度优化耗时: {depth_optimized_et-depth_optimized_st:.4f} 秒")

    # cv2.imwrite(data_dir+"depth_optimized.png", depth_optimized)
    loaded_depth_optimized = cv2.imread(data_dir + "depth_optimized.png", cv2.IMREAD_ANYDEPTH) 
    #show_image(loaded_depth_optimized)

    #可视化修正的场景点云
    
    #create_pointcloud_from_rgbd(intrinsic = config.cam_info["intrinsic"], color_img = loaded_color_rgb, depth_img = depth_infer)
    
    #获取分割
    #targer_mask = get_target_mask(loaded_color_rgb,user_input,sammodel)
    #save_mask_as_image(targer_mask,data_dir+"mask.png")
    loaded_mask = cv2.imread(data_dir + "mask.png")
    #show_image(loaded_mask)
    #生成可行抓取姿态
    viable_graspgroup, near_joint_angle_list, grasp_joint_angle_list, cloud = find_viable_grasps(
        piper,
        graspnet, 
        loaded_color_rgb, 
        loaded_depth_optimized, 
        loaded_mask, 
        config,
        max_retries=100
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



    piper.control_joint([-0.675, 23.462, -63.198, 2.623, 72.915, 0.651])
    time.sleep(3)
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

    print("回到预抓取点")
    piper.control_joint([-0.675, 23.462, -63.198, 2.623, 72.915, 0.651])
    #piper.control_joint([-1.0257399367606437, 67.22925267717845, -66.29589135109437, 2.506766289794974, 64.5593426076473, -1.9640179457439293])
    time.sleep(2)

    # # 抬起路径点1
    # piper.control_joint([1.6789470513973734, 119.87612682338631, -82.0777912024451, -20.921370349870255, 31.802037307480244, 38.17365898204937])
    # time.sleep(0.6)
    # # 抬起路径点2
    # piper.control_joint([-0.675, 56.035, -54.772, 3.18, 48.839, 5.661])
    # time.sleep(0.6)
    # # 抬起路径点3
    # piper.control_joint([-0.675, 23.462, -63.198, 2.623, 72.915, 0.651])
    # time.sleep(0.6)




    #关闭外设
    cam.stop()
    piper.disconnect()

if __name__ == "__main__":
    main()