import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="./assets/boat.png", help="path to the image")
parser.add_argument("--output_dir", type=str, default="./test/boat", help="path to the output directory")
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/trellis_oa", help="path to the checkpoint directory")
args = parser.parse_args()

# Load a pipeline from a model folder.
pipeline = TrellisImageTo3DPipeline.from_pretrained(args.ckpt_dir)
pipeline.cuda()

output_dir = os.path.join(args.output_dir, "trellis_oa_results")
os.makedirs(output_dir, exist_ok=True)
# Load an image
image = Image.open(args.image_path)

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 3.0,
    },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
# video = render_utils.render_video(outputs['gaussian'][0])['color']
# imageio.mimsave(os.path.join(args.output_dir, "sample_gs.mp4"), video, fps=30)
# video = render_utils.render_video(outputs['radiance_field'][0])['color']
# imageio.mimsave(os.path.join(args.output_dir, "sample_rf.mp4"), video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['color']
imageio.mimsave(os.path.join(output_dir, "sample_mesh.mp4"), video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(os.path.join(output_dir, "sample.glb"))

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(os.path.join(output_dir, "sample.ply"))
