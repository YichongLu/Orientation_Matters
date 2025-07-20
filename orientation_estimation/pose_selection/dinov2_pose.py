import argparse
import torch
from pathlib import Path
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import torch.nn as nn
import os
import time
import shutil
from sklearn.decomposition import PCA
import json
import cv2
import torchvision.transforms as T

def project_3d_to_2d(pt,K,ob_in_cam):
    pt = pt.reshape(4,1)
    projected = K @ ((ob_in_cam@pt)[:3,:])
    projected = projected.reshape(-1)
    projected = projected/projected[2]
    return projected.reshape(-1)[:2].round().astype(int)

def draw_arrow(img, start_point, end_point, color, thickness=3):

    cv2.arrowedLine(img, start_point, end_point, color, thickness,tipLength=0)
    triangle_end_point = (int(end_point[0]*1.006), int(end_point[1]*1.006))

    # triangle_end_point = end_point
    # 计算箭头的长度和方向
    arrow_length = int(512/50)
    angle = np.arctan2(triangle_end_point[1] - start_point[1], triangle_end_point[0] - start_point[0])

    # 计算箭头的两个侧边点
    arrow_tip1 = (triangle_end_point[0] - arrow_length * np.cos(angle - np.pi / 6), triangle_end_point[1] - arrow_length * np.sin(angle - np.pi / 6))
    arrow_tip2 = (triangle_end_point[0] - arrow_length * np.cos(angle + np.pi / 6), triangle_end_point[1] - arrow_length * np.sin(angle + np.pi / 6))

    # 创建一个封闭多边形作为箭头头部
    arrow_pts = np.array([triangle_end_point, arrow_tip1, arrow_tip2], np.int32)
    arrow_pts = arrow_pts.reshape((-1, 1, 2))

    # 使用 cv2.fillPoly 绘制实心箭头
    cv2.fillPoly(img, [arrow_pts], color,lineType = cv2.LINE_AA)
    
    # # 在箭头末端（终点）附近添加文本
    # text = "Label"
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.8
    # thickness = 2
    # text_color = color  # 黑色文本

    # # 计算文本大小，调整位置（避免遮挡箭头）
    # (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # text_x = end_point[0] - text_width // 2  # 水平居中
    # text_y = end_point[1] - 10  # 在箭头上方

    # # 添加文本
    # cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    
    return img

def draw_xyz_axis(color, ob_in_cam, scale=0.5, K=np.eye(3), thickness=3, transparency=0,is_input_rgb=False):
    '''
    @color: BGR
    '''
    if is_input_rgb:
        color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
    #   origin = (color.shape[1]//2, color.shape[0]//2)
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0.2
    # ... (计算 xx,yy,zz)
    color = draw_arrow(color, origin, xx, (0,0,255), thickness) # X轴红色
    color = draw_arrow(color, origin, yy, (0,255,0), thickness) # Y轴绿色
    color = draw_arrow(color, origin, zz, (255,0,0), thickness) # Z轴蓝色
    # if is_input_rgb:
    #     color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    # tmp = color.copy()
    # tmp1 = tmp.copy()
    # tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    # mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    # tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    # tmp1 = tmp.copy()
    # tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    # mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    # tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    # tmp1 = tmp.copy()
    # tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    # mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    # tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    # tmp = tmp.astype(np.uint8)
    # if is_input_rgb:
    #     tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

    return color

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    # x.reshape(2,1,int(x.shape[2]/2),x.shape[3])
    # y.reshape(2,1,int(y.shape[2]/2), y.shape[3])
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

def find_similarity_mse(image_path1: str, image_path2: str, extractor , idx, num_pairs: int = 8, load_size: int = 112, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 2) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                                                              Image.Image, Image.Image]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """

    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size
    
    # np.save("/home/jxhuang/others/dino-vit-features-main/test/feature_map_ref_5994.npz", descriptors1.squeeze().cpu().numpy().reshape((num_patches1[0], num_patches1[1], 6528)))
    # np.save("/home/jxhuang/others/dino-vit-features-main/test/feature_map_cad_5994.npz", descriptors2.squeeze().cpu().numpy().reshape((num_patches2[0], num_patches2[1], 6528)))

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]

    mse_loss = nn.MSELoss()
    mse_similarity = 1 - mse_loss(descriptors1, descriptors2)
    print('renderding_idx :' + str(idx) + ' similarity : ' + str(mse_similarity))
    return mse_similarity, descriptors1, num_patches1, descriptors2, num_patches2



def uniform_search(num_frames, load_size, cad_rgb_path, save_dir, cad_name):
    # uniform pose searching
    points1_list = []
    points2_list = []
    image1_list = []
    image2_list = []
            
    for idx in range(num_frames):
        # cad_image_path = os.path.join(output_path, scene_name, object_name, 'rgb','%06d_0001.png' % idx)
        cad_image_path = os.path.join(cad_rgb_path, '%06d_0001.png' % idx)
        # similarity = find_similarity_mse(ref_image_path, cad_image_path, extractor, idx)
        similarity, points1, points2, image1_pil, image2_pil = find_similarity_pointbased(ref_image_path, cad_image_path, extractor, idx, load_size=load_size)
        # points1_list.append(points1)
        # points2_list.append(points2)
        # image1_list.append(image1_pil)
        # image2_list.append(image2_pil)
        similarity_list.append(similarity)
    
    # set_trace()
        
    similarity_max = max(similarity_list)
    rendering_id = similarity_list.index(similarity_max)
    # fig1, fig2 = draw_correspondences(points1_list[rendering_id], points2_list[rendering_id], image1_list[rendering_id], image2_list[rendering_id])        
    # fig1.savefig(os.path.join(save_dir, f'{cad_name}_ref.png'), bbox_inches='tight', pad_inches=0)
    # fig2.savefig(os.path.join(save_dir, cad_name+'.png'), bbox_inches='tight', pad_inches=0)
    # plt.close('all')
    
    return rendering_id, similarity_list

def l2_search(num_frames, dinov2_model, input_dir, output_dir):
    similarity_list = []    
    for idx in range(num_frames):
        rgbA_path = os.path.join(input_dir, 'rgbA_vis_%d.png' % idx)
        rgbB_path = os.path.join(input_dir, 'rgbB_vis_%d.png' % idx)
        
        rgbA = Image.open(rgbA_path).convert("RGB")
        rgbB = Image.open(rgbB_path).convert("RGB")
        
        image_transforms = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        rgbA = image_transforms(rgbA)[:3].unsqueeze(0).cuda()
        rgbB = image_transforms(rgbB)[:3].unsqueeze(0).cuda()
        
        with torch.no_grad():
            featuresA = dinov2_model.forward_features(rgbA)["x_norm_patchtokens"]
            featuresB = dinov2_model.forward_features(rgbB)["x_norm_patchtokens"]
            mse_loss = nn.MSELoss()
            mse_similarity = 1 - mse_loss(featuresA, featuresB)
            print('renderding_idx :' + str(idx) + ' similarity : ' + str(mse_similarity))
        
        # similarity, descriptors1, num_patches1, descriptors2, num_patches2 = find_similarity_mse(rgbA, rgbB, extractor, idx, load_size=load_size)
            
        # pca = PCA(n_components=3)
        # descriptors1_copy = descriptors1.detach().cpu().numpy().copy().squeeze()
        # descriptors2_copy = descriptors2.detach().cpu().numpy().copy().squeeze()
        # pca.fit(descriptors2_copy)
        
        # projected_des1 = pca.transform(descriptors1_copy)
        # t = torch.tensor(projected_des1)
        # t_min = t.min(dim=0, keepdim=True).values
        # t_max = t.max(dim=0, keepdim=True).values
        # normalized_t = (t - t_min) / (t_max - t_min)

        # array1 = (normalized_t * 255).byte().numpy()
        # array1 = array1.reshape((num_patches2[0], num_patches2[1],3))
        
        # projected_des2 = pca.transform(descriptors2_copy)
        # t = torch.tensor(projected_des2)
        # t_min = t.min(dim=0, keepdim=True).values
        # t_max = t.max(dim=0, keepdim=True).values
        # normalized_t = (t - t_min) / (t_max - t_min)

        # array2 = (normalized_t * 255).byte().numpy()
        # array2 = array2.reshape((num_patches2[0], num_patches2[1],3))
        
        # Image.fromarray(array1).resize((200,200)).save(os.path.join(output_dir, f'{idx}_target_feature.png'))
        # Image.fromarray(array2).resize((200,200)).save(os.path.join(output_dir, 'reference_feature.png'))
        
        similarity_list.append(mse_similarity)
    
    similarity_max = max(similarity_list)
    rendering_id = similarity_list.index(similarity_max)
    print("Rendering_idx : " + str(rendering_id))
    return rendering_id, similarity_list

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir.')
    parser.add_argument('--test_image', type=str, required=True, help='The test image.')
    parser.add_argument('--load_size', default=112, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=8, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=9, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='True', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--thresh', default=0.05, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--num_pairs', default=10, type=int, help='Final number of correspondences.')
    parser.add_argument('--num_frames', default=252, type=int, help='Number of CAD rendering frames.')
    parser.add_argument('--match_type', default='l2', type=str)
    parser.add_argument('--model_size', default='large', type=str)
    parser.add_argument("--ckpt_root_dir", type=str, required=True)
    args = parser.parse_args()

    with torch.no_grad():
        
        # prepare directories
        input_dir = os.path.join(args.root_dir, "foundationpose_outputs")
        output_dir = os.path.join(args.root_dir, "orientation_estimation_results")
        num_frames = args.num_frames
        load_size = args.load_size
        match_type = args.match_type
        refiner_dir = os.path.join(input_dir, 'refiner')
        
        if os.path.exists(output_dir):
            os.system(f"rm -rf {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # extracting descriptors for each image
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dinov2_model = torch.hub.load(os.path.join(args.ckpt_root_dir, 'pose_selection', 'facebookresearch_dinov2_main'), 'dinov2_vitl14_reg', source='local')
        dinov2_model.eval().cuda()
        
        rendering_id, similarity_list = l2_search(num_frames, dinov2_model, refiner_dir, output_dir)

        with open(os.path.join(output_dir, 'rendering_id.json'), 'w') as f:
            json.dump(str(rendering_id), f)
        
        os.system(f"cp {os.path.join(refiner_dir,'rgbA_vis_%d.png' % rendering_id)} {os.path.join(output_dir, 'rgbA_vis.png')}")
        os.system(f"cp {os.path.join(refiner_dir,'rgbB_vis_%d.png' % rendering_id)} {os.path.join(output_dir, 'rgbB_vis.png')}")
        os.system(f"cp {os.path.join(refiner_dir,'ob_in_cam_%d.txt' % rendering_id)} {os.path.join(output_dir, 'ob_in_cam.txt')}")
        
        with open(os.path.join(output_dir, 'ob_in_cam.txt'), 'r') as f:
            ob_in_cam = np.loadtxt(f)
            
        with open(os.path.join(refiner_dir,'K.txt'), 'r') as f:
            K = np.loadtxt(f)
                
        color_rgb = np.array(Image.open(args.test_image))[:,:,:3]
        # color_bgr = color_rgb[:,:,::-1]
        
        vis = draw_xyz_axis(color_rgb, ob_in_cam=ob_in_cam, scale=0.5, K=K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imwrite(os.path.join(output_dir, 'vis.png'), vis)
        
        similarity_list_str = []
        for similarity in similarity_list:
            similarity_list_str.append(str(similarity))

        with open(os.path.join(output_dir, 'similarity_list.json'), 'w') as f:
            json.dump(similarity_list_str, f)
 