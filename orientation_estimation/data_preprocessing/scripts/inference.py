from unidepth.models import UniDepthV1
import numpy as np
from PIL import Image
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_root_dir", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    output_dir = args.output_dir
    model = UniDepthV1.from_pretrained(os.path.join(args.ckpt_root_dir, "data_preprocessing")) # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone

    output_dir = os.path.join(output_dir, "foundationpose_inputs")
    
    # Move to CUDA, if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the RGB image and the normalization will be taken care of by the model
    rgb = torch.from_numpy(np.array(Image.open(image_path)))[..., :3].permute(2, 0, 1) # C, H, W

    predictions = model.infer(rgb)

    # Metric Depth Estimation
    depth = predictions["depth"]

    # Point Cloud in Camera Coordinate
    xyz = predictions["points"]

    # Intrinsics Prediction
    intrinsics = predictions["intrinsics"]

    # Save the depth map
    depth_map = (depth.squeeze().cpu().numpy() * 1000.0).astype(np.uint16)
    image_name = image_path.split("/")[-1]
    depth_output_path = os.path.join(output_dir, 'depth')
    os.makedirs(depth_output_path, exist_ok=True)
    Image.fromarray(depth_map).save(os.path.join(depth_output_path, image_name))
    
    # Save the intrinsics
    cam_K_output_path = output_dir
    with open(os.path.join(cam_K_output_path, "cam_K.txt"), "w") as f:
        f.write("%f %f %f\n" % (intrinsics[0, 0, 0], intrinsics[0, 0, 1], intrinsics[0, 0, 2]))
        f.write("%f %f %f\n" % (intrinsics[0, 1, 0], intrinsics[0, 1, 1], intrinsics[0, 1, 2]))
        f.write("%f %f %f\n" % (intrinsics[0, 2, 0], intrinsics[0, 2, 1], intrinsics[0, 2, 2]))


