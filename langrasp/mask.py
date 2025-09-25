"""
用于从rgb中生成给定指令的mask
"""
import os
import io
import base64
import cv2
import torch
import json
from PIL import Image
import numpy as np

from openai import OpenAI
client = OpenAI(
    api_key="sk-73fe202ace944ca6864986ef41dbc72d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def image_to_base64(image):
    """
    将 PIL.Image 或 numpy.ndarray 图像对象编码为 base64 字符串（JPEG 格式）
    自动检测输入类型并转换。
    支持：
        - PIL.Image.Image 对象
        - numpy.ndarray (H, W, 3) 或 (H, W) —— 自动转 RGB 或灰度
    """
    # 如果是 numpy 数组
    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # 灰度图
            pil_image = Image.fromarray(image, mode='L')
        elif image.ndim == 3:
            if image.shape[2] == 3:  # RGB
                pil_image = Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(image, mode='RGBA').convert('RGB')  # 转为 RGB 以兼容 JPEG
            else:
                raise ValueError(f"Unsupported channel number: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {image.shape}")
    # 如果是 PIL Image
    elif hasattr(image, 'save') and hasattr(image, 'convert'):  # 简单判断是否为 PIL 图像
        pil_image = image.convert('RGB')  # 确保是 RGB，避免 RGBA 保存 JPEG 报错
    else:
        raise TypeError("Input must be a PIL.Image.Image or numpy.ndarray")

    # 编码为 JPEG base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)  # 可选 quality 控制压缩质量
    image_bytes = buffer.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image

def get_prediction_result_qwen(image, sammodel, scene_id, obj_id, text):
    """
    1. 使用gpt来推理下一步应抓取的物体box和points
    2. 根据传入的实例化sammodel和points来生成mask
    3. 得到目标物体mask
    """
    save_dir = "/root/host_share/langrasp/data"
    os.makedirs(save_dir, exist_ok=True)

    max_retries = 3  # 最大重试次数
    retry_count = 0
    system_prompt = """
        想象你是一个用于机械臂抓取的感知系统，你的任务是在杂乱场景下进行抓取，你需要根据user输入确定当前需要抓取的物体，规则是这样:
        1.如果用户期望的目标物体没有被遮挡，返回该物体上的点和类别标签
        2.如果用户期望的目标物体被遮挡，那么识别一个遮挡目标且本身没有障碍物的物体，返回遮挡目标的点和类别标签

        你需要返回物体的边界框和物体上的3个点
        输出格式:
        {
        "points": [
        [234, 45],
        [256, 78],
        [289, 56]
        ],
        "bbox": [x_min, y_min, x_max, y_max],
        "label": "cup",
        "is_occluded": false
        }
        """
    while retry_count < max_retries:
        try:
            # Load image as base64
            base64_image = image_to_base64(image)
            # GPT input text
            input_text = f"Grasp {text}"

            messages = [
                {
                "role": "system",
                "content": system_prompt
                                
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]

            response = client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=messages,
                # temperature=0,
                # max_tokens=713,
            )

            output = response.choices[0].message.content
            print(f"User Input: {input_text}")
            print("GPT Output:", output)
            
            # 使用json模块解析
            result = json.loads(output)
            points = result["points"]
            bbox = result["bbox"]
            label = result["label"]
            is_occluded = result["is_occluded"]

            # 对points和bbox进行坐标偏移修正
            x_fix = 0
            y_fix = 0
            adjusted_points = []
            for point in points:
                adjusted_points.append([point[0] + x_fix, point[1] + y_fix])

            adjusted_bbox = [
                bbox[0] + x_fix,  # x_min + 50
                bbox[1] + y_fix, # y_min + 100
                bbox[2] + x_fix,  # x_max + 50
                bbox[3] + y_fix  # y_max + 100
            ]

            # 使用 points 提示SAM
            results = sammodel(image, device=0, retina_masks=True, conf=0.72, iou = 0.8,points=adjusted_points, labels=[1]*len(points))
            results = sammodel(image, bboxes=adjusted_bbox)
            # 可视化VLM给出的points和bbox
            annotated_img = results[0].plot()
            for point in adjusted_points:
                x_int, y_int = int(point[0]), int(point[1])
                cv2.circle(annotated_img, (x_int, y_int), 5, (0, 0, 255), 2)
                cv2.putText(annotated_img, f"({x_int},{y_int})",
                            (x_int + 10, y_int - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            x_min, y_min, x_max, y_max = map(int, adjusted_bbox)
            cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 绿色框
            cv2.putText(annotated_img, f"VLM bbox: {label}", 
                        (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                

            save_path = os.path.join(save_dir, f"{scene_id}__{obj_id}_{text}_SamGpt.jpg")
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, annotated_img_rgb)

            masks = results[0].masks.data.cpu().numpy()
            h, w = masks.shape[1:]

            # 统计每个mask包含多少点，并计算面积
            mask_info = []
            for i, mask in enumerate(masks):
                count = 0
                for (x, y) in adjusted_points:
                    xi, yi = int(x), int(y)
                    if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0.5:
                        count += 1
                area = mask.sum()  # 面积
                #print(i, count, area)
                mask_info.append((i, count, area))

            # 先按点数排序，再按面积排序
            best_mask_idx, _, _ = max(mask_info, key=lambda x: (x[1], -x[2]))

            # 获取最佳mask并转torch.bool
            best_mask = masks[best_mask_idx] > 0.5
            

            return best_mask

        except Exception as e:
            print(f"第 {retry_count + 1} 次尝试失败: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                return {"error": f"重试{max_retries}次后仍失败: {str(e)}"}, {}

    # 如果循环结束仍未返回，说明失败
    return {"error": "未能生成有效 mask，重试次数耗尽"}, {}

def save_mask_as_image(mask, output_path, format='PNG', threshold=0.5):
    """
    将布尔类型或浮点(0.0/1.0)的NumPy数组保存为图片
    
    Args:
        mask: 布尔类型或浮点(0.0/1.0)的NumPy数组
        output_path: 输出图片的路径
        format: 图片格式，如'PNG', 'JPEG', 'BMP'等
        threshold: 浮点值转换为布尔值的阈值，默认为0.5
    """
    # 检查输入是否为NumPy数组
    if not isinstance(mask, np.ndarray):
        raise ValueError("输入必须是NumPy数组")
    
    # 处理不同的数据类型
    if mask.dtype == bool:
        # 如果已经是布尔类型，直接使用
        image_array = np.where(mask, 255, 0).astype(np.uint8)
    elif np.issubdtype(mask.dtype, np.floating):
        # 如果是浮点类型，使用阈值转换为布尔类型
        # 大于等于阈值的为True(255)，小于阈值的为False(0)
        image_array = np.where(mask >= threshold, 255, 0).astype(np.uint8)
    else:
        raise ValueError("不支持的数据类型，仅支持布尔类型或浮点类型")
    # 创建PIL图像对象
    img = Image.fromarray(image_array)
    
    # 保存图像
    img.save(output_path, format=format)

def get_target_mask(image_rgb, user_input,sammodel):
    """
    根据用户输入和场景图像，生成目标物体的掩码
    """
    scene_id = 1
    obj_id = 1
    return get_prediction_result_qwen(image_rgb, sammodel, scene_id, obj_id, user_input)
    