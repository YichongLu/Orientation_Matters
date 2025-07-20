import os
import shutil
from PIL import Image

rgb_dir = "/data2/yclu/unidepth/inputs/gso_30"
depth_dir = "/data2/yclu/unidepth/outputs/gso_30/depth"
mesh_dir = "/data0/yclu/wonder3d/evaluation/GT/gso"
cam_K_dir = "/data2/yclu/unidepth/outputs/gso_30/cam_K"

target_root_dir = "/data2/yclu/foundationpose/inputs/gso_30"
if os.path.exists(target_root_dir):
    shutil.rmtree(target_root_dir)
os.makedirs(target_root_dir, exist_ok=True)

obj_names = os.listdir(rgb_dir)

for obj_name in obj_names:
    rgb_path = os.path.join(rgb_dir, obj_name)
    depth_path = os.path.join(depth_dir, obj_name)
    mesh_obj_path = os.path.join(mesh_dir, obj_name.split(".")[0], 'meshes', 'model.obj')
    mesh_mtl_path = os.path.join(mesh_dir, obj_name.split(".")[0], 'meshes', 'model.mtl')
    mesh_texture_path = os.path.join(mesh_dir, obj_name.split(".")[0], 'meshes', 'texture.png')
    cam_K_path = os.path.join(cam_K_dir, obj_name.split(".")[0] + '_cam_K.txt')

    rgb_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'rgb', obj_name)
    depth_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'depth', obj_name)
    mesh_obj_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'mesh', 'model.obj')
    mesh_mtl_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'mesh', 'model.mtl')
    mesh_texture_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'mesh', 'texture.png')
    cam_K_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'cam_K.txt')

    # Load the RGBA image
    rgba_image = Image.open(rgb_path)
    # Split the RGBA image into individual bands
    r, g, b, a = rgba_image.split()
    # Convert the alpha channel to a 3-channel PNG mask
    mask_image = a.convert('RGB')
    # Save the mask image
    mask_target_path = os.path.join(target_root_dir, obj_name.split(".")[0], 'masks', obj_name)
    os.makedirs(os.path.dirname(mask_target_path), exist_ok=True)
    mask_image.save(mask_target_path)
    
    os.makedirs(os.path.dirname(rgb_target_path), exist_ok=True)
    os.makedirs(os.path.dirname(depth_target_path), exist_ok=True)
    os.makedirs(os.path.dirname(mesh_obj_target_path), exist_ok=True)
    os.makedirs(os.path.dirname(mesh_mtl_target_path), exist_ok=True)
    os.makedirs(os.path.dirname(mesh_texture_target_path), exist_ok=True)
    os.makedirs(os.path.dirname(cam_K_target_path), exist_ok=True)

    os.system(f"cp {rgb_path} {rgb_target_path}")
    os.system(f"cp {depth_path} {depth_target_path}")
    os.system(f"cp {mesh_obj_path} {mesh_obj_target_path}")
    os.system(f"cp {mesh_mtl_path} {mesh_mtl_target_path}")
    os.system(f"cp {mesh_texture_path} {mesh_texture_target_path}")
    os.system(f"cp {cam_K_path} {cam_K_target_path}")
