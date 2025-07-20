import cv2
import numpy as np


def depth_to_vis(depth, zmin=None, zmax=None, mode='rgb', inverse=True):
    if zmin is None:
        zmin = depth.min()
    if zmax is None:
        zmax = depth.max()

    if inverse:
        invalid = depth<0.001
        vis = zmin/(depth+1e-8)
        vis[invalid] = 0
    else:
        depth = depth.clip(zmin, zmax)
        invalid = (depth==zmin) | (depth==zmax)
        vis = (depth-zmin)/(zmax-zmin)
        vis[invalid] = 1

    if mode=='gray':
        vis = (vis*255).clip(0, 255).astype(np.uint8)
    elif mode=='rgb':
        vis = cv2.applyColorMap((vis*255).astype(np.uint8), cv2.COLORMAP_JET)[...,::-1]
    else:
        raise RuntimeError

    return vis