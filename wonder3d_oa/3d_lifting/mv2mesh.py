
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import time
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
from PIL import Image

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# # load image dream
# pipe = MVDreamPipeline.from_pretrained(
#     "ashawkey/imagedream-ipmv-diffusers", # remote weights
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     # local_files_only=True,
# )
# pipe = pipe.to(device)

# # load rembg
# bg_remover = rembg.new_session()

# process function
def process(opt: Options, path):
    name = path.split('/')[-1]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.save_dir, exist_ok=True)

    front_image = (np.array(Image.open(os.path.join(path,'rgb_000_front.png')), dtype=np.float32)/255.0)
    if front_image.shape[-1] == 4:
        front_image = front_image[..., :3] * front_image[..., 3:4] + (1 - front_image[..., 3:4])
    right_image = (np.array(Image.open(os.path.join(path,'rgb_000_right.png')), dtype=np.float32)/255.0)
    if right_image.shape[-1] == 4:
        right_image = right_image[..., :3] * right_image[..., 3:4] + (1 - right_image[..., 3:4])
    back_image = (np.array(Image.open(os.path.join(path,'rgb_000_back.png')), dtype=np.float32)/255.0)
    if back_image.shape[-1] == 4:
        back_image = back_image[..., :3] * back_image[..., 3:4] + (1 - back_image[..., 3:4])
    left_image = (np.array(Image.open(os.path.join(path,'rgb_000_left.png')), dtype=np.float32)/255.0)
    if left_image.shape[-1] == 4:
        left_image = left_image[..., :3] * left_image[..., 3:4] + (1 - left_image[..., 3:4])
    
    mv_image = np.stack([front_image, right_image, back_image, left_image], axis=0)

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.save_dir, 'sample.ply'))

        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        # imageio.mimwrite(os.path.join(opt.save_dir, 'sample.mp4'), images, fps=30)


if opt.input_path:
    file_paths = [opt.input_path]
elif opt.input_dir:
    file_paths = glob.glob(os.path.join(opt.input_dir, "*"))
else:
    raise ValueError(f'input_path or input_dir must be provided')

for path in file_paths:
    process(opt, path)
