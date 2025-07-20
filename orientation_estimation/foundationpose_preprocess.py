import os
import cv2
import numpy as np
import rembg
import kiui
import argparse
from rembg import remove

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/data2/yclu/oa3d/pose_inputs/test/elephant_rotated.png")
    parser.add_argument("--output_dir", type=str, default="/data2/yclu/oa3d/pose_inputs/test/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, "foundationpose_inputs")
    
    image = cv2.imread(image_path)
    
    mask_dir = os.path.join(output_dir, "masks")
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    rm_pred = remove(image)
    mask_pred = rm_pred[..., -1]
    mask_pred = np.stack((mask_pred, mask_pred, mask_pred), axis=-1)
    cv2.imwrite(os.path.join(mask_dir, os.path.basename(image_path)), mask_pred)

    image = image * (mask_pred/255)
    cv2.imwrite(os.path.join(rgb_dir, os.path.basename(image_path)), image)
