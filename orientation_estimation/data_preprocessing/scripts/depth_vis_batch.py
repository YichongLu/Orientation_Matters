import os
import cv2
import numpy as np
import sys
sys.path.append("./scripts")
from utils import depth_to_vis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--depth_dir', type=str, default='/data2/yclu/unidepth/outputs/gso_30/depth')
parser.add_argument('--output_dir', type=str, default='/data2/yclu/unidepth/outputs/gso_30/depth_vis')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

depth_files = os.listdir(args.depth_dir)

for depth_file in depth_files:
    depth = cv2.imread(os.path.join(args.depth_dir, depth_file), cv2.IMREAD_ANYDEPTH)
    H,W = depth.shape[:2]
    zmin = depth.min()
    zmax = depth.max()

    depth_vis = depth_to_vis(depth, zmin=zmin, zmax=zmax, inverse=False, mode='gray')
    cv2.imwrite(os.path.join(args.output_dir, depth_file), depth_vis)
