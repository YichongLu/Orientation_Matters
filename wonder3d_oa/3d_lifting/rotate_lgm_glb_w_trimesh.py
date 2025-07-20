import trimesh
import numpy as np
import trimesh.transformations as tf
import os
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

if not args.model_path.endswith(".glb"):
    raise ValueError("File must be a glb file")

mesh = trimesh.load_mesh(args.model_path)

rotation_matrix = tf.rotation_matrix(np.radians(90), [0, 1, 0])

mesh.apply_transform(rotation_matrix)

mesh.export(args.model_path)

