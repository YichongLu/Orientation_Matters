import os
import numpy as np
import cv2
import random
import argparse
from ipdb import set_trace
import shutil
import json

def create_static_color(image, output_path):
    color_dict = dict()
    for i in range(300):
        color_dict[i] = random_color()
    
    with open(os.path.join(output_path, 'static_color.json'), 'w') as json_file:
        json.dump(color_dict, json_file)
    
    return color_dict
    
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def color_segmentation(image_path, output_path, color_mapping_path, num_label_dir):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read {image_path}")
        return

    # 创建彩色图层
    colored_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8)*255

    # 创建颜色映射字典
    color_mapping = {}
    
    color_dict_json_path = os.path.join(output_path, 'static_color.json')
    
    if not os.path.exists(color_dict_json_path):
    
        create_static_color(image, output_path)
    
    # set_trace()
    
    with open(color_dict_json_path, 'r') as json_file:
        color_dict = json.load(json_file)
    
    # 为每个唯一的像素值（即分割区域）上色
    for label in np.unique(image):
        # set_trace()
        if label == 0:  # 假设0是背景
            continue
        # if label != 1 & label != 9:
        #     continue
        mask = image == label
        # choose random color
        # color = random_color()
        # choose static color
        color = color_dict[str(label)]
        colored_image[mask] = color
        color_mapping[label] = color

    # 保存着色后的图像
    colored_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(colored_image_path, colored_image)
    print(f"Image saved to {colored_image_path}")

    # 保存颜色映射
    # mapping_file = os.path.join(color_mapping_path, os.path.basename(image_path) + ".txt")
    # with open(mapping_file, 'w') as file:
    #     for label, color in color_mapping.items():
    #         file.write(f"{label}: {color}\n")
            
    # print(f"Color mapping saved to {mapping_file}")
    
    ### xt
    # 保存标签可视化图像
    # output_image = colored_image.copy()
    # visual_mask = os.path.join(num_label_dir, os.path.basename(image_path))
    # print('number visual img', visual_mask)
    # color_step = 255 / len(color_mapping)
    # for label, color in color_mapping.items():
    #     try:
    #         mask = cv2.inRange(output_image, color, color)
            
    #         # _, connected_label, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #         # for conct_label in range(1, len(stats)):
    #         # x, y, w, h, count = stats[0]
    #         center_y = mask.nonzero()[0][mask.nonzero()[1].argmin()]
    #         center_x = mask.nonzero()[1][mask.nonzero()[1].argmin()]
    #         thickness = 2# if count < 50 else 1  ##数量较少的加粗一下
    #             # cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #         cv2.putText(output_image, str(label), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , thickness)
    #     except:
    #         continue
    # cv2.imwrite(visual_mask, output_image)
    # print(f"map of labels and RGB colors saved to {visual_mask}")
    ###
            
       

parser = argparse.ArgumentParser(description='Material Optimization Script for PhotoScene')
parser.add_argument('--inp_path', required=True)
args = parser.parse_args()
# input_dir = args.inp_path #'/data0/zsz/PhotoScene/outputs/01926_001_final_results0001/seg_merge'
input_dir = args.inp_path#r'/public2/home/xutao_meinv/project/photoscene_new/data/01926_001_final_results0001/seg_merge'
replace_name = args.inp_path.split('/')[-1]
output_dir = input_dir.replace(replace_name, replace_name+'_colored')
color_mapping_dir = input_dir.replace(replace_name, replace_name+'_color_mapping')

### xt
num_label_dir = input_dir.replace(replace_name, replace_name+'_num_mapping')
os.makedirs(num_label_dir, exist_ok=True)
###

# 确保输出目录和颜色映射目录存在

# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
# os.makedirs(output_dir)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(color_mapping_dir, exist_ok=True)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    if os.path.isfile(file_path):
        color_segmentation(file_path, output_dir, color_mapping_dir, num_label_dir)
