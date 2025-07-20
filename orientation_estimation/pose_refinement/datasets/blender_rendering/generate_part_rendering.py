"""
blender renderer
"""
import argparse
import bpy
import copy
import math
import mathutils
import numpy as np
import os 
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.dirname(__file__))
import util
from math import degrees, radians
from sklearn.cluster import KMeans
from ipdb import set_trace
from PIL import Image
import shutil


def import_mesh(fpath, scale=1., object_world_matrix=None):
        bpy.ops.wm.obj_import(filepath=str(fpath), use_split_objects=False)
        obj = bpy.context.selected_objects[0]
        util.dump(bpy.context.selected_objects)
        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0., 0., 0.) # center the bounding box!

        if scale != 1.:
            bpy.ops.transform.resize(value=(scale, scale, scale))


def get_archimedean_spiral(sphere_radius, num_steps=250):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 2
    r = sphere_radius
    i_list = np.linspace(-a / 2, a/2, num_steps)
    translations = []
    # mids = .5 * (i_list[...,1:] + i_list[...,:-1])
    # upper = np.concatenate([mids, i_list[...,-1:]], -1)
    # lower = np.concatenate([i_list[...,:1], mids], -1)
    #torch.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    # t_rand = np.random.rand(*i_list.shape)
    # i_list = lower + (upper - lower) * t_rand
    
    for i in i_list:
        rt = r 
        theta =  math.pi * 1 / 2 + math.pi * 1 / 8
        cos_v = math.cos(-i *  math.pi)
        sin_v = math.sin(-i *  math.pi)
        x = rt * math.sin(theta) * cos_v
        z = rt * math.sin(-theta + math.pi) * sin_v
        y = rt * - math.cos(theta)

        translations.append((x, y, z))

    return np.array(translations)

def point_at(obj, target, roll=0):
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)
    loc = obj.location
    direction = target - loc
    quat = direction.to_track_quat('-Z', 'Y')
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, 'Z')
    loc = loc.to_tuple()
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc

def get_materials_metallic():
    mats = bpy.data.materials
    materials_metallic = []
    for m in mats:
        materials_metallic.append(m.metallic)
    return materials_metallic

def get_materials_roughness():
    mats = bpy.data.materials
    materials_roughness = []
    for m in mats:
        materials_roughness.append(m.roughness)
    return materials_roughness

def get_materials_color():
    mats = bpy.data.materials
    materials_color = []
    for m in mats:
        materials_color.append(m.diffuse_color)
    return materials_color

def get_materials_names():
    mats = bpy.data.materials
    materials_names = []
    for m in mats:
        materials_names.append(m.name)
    return materials_names

def new_color_material(mat_name, color, roughness, metallic, shadow_mode='NONE'):
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    # principled_node = nodes.get('Principled BSDF')
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.get("Material Output")
    principled_node.inputs.get("Base Color").default_value = color
    principled_node.inputs.get("Alpha").default_value = color[3]
    principled_node.inputs.get("Roughness").default_value = roughness
    principled_node.inputs.get("Metallic").default_value = metallic
    # principled_node.inputs.get("Emission Strength").default_value = 0
    link = links.new( principled_node.outputs['BSDF'], output_node.inputs['Surface'] )
    if color[-1] < 1:
        mat.blend_method = 'BLEND'
    mat.shadow_method = shadow_mode
    return mat

def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * math.cos(azimuth) * math.sin(elevation)
    y = radius * math.cos(elevation)
    z = radius * math.sin(azimuth) * math.sin(elevation)
    x_blender = x * 1.15
    y_blender = -z * 1.15
    z_blender = y * 1.15
    return (x_blender, y_blender, z_blender)

def set_uniform_lights():
    focus_point = [0,0,0]
    light_distance = 5
    light_names = ['Light_front', 'Light_back', 'Light_left', 'Light_right', 'Light_top', 'Light_bottom']
    light_locations = []
    for i in range(3):
        light_location = focus_point[:]
        light_location[i] -= light_distance
        light_locations.append(light_location)
        light_location = focus_point[:]
        light_location[i] += light_distance
        light_locations.append(light_location)
        
    for i in range(len(light_names)):
        light_data = bpy.data.lights.new(name=light_names[i], type='POINT')
        light_data.energy = 500
        light_object = bpy.data.objects.new(name=light_names[i], object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object
        light_object.location = light_locations[i]

def set_gpu_renderer():
    bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.display.render_aa = 'OFF'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        # only use GPU.
        if 'GPU' not in d["name"]:
            d["use"] = 0 
        else:
            d["use"] = 1
        d["use"] = 1
        print(d["name"], d["use"])
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.view_settings.view_transform = 'Standard'
    num_samples = 64
    bpy.context.scene.cycles.samples = num_samples

def set_resolution(H=256, W=256):

    num_samples = 2048

    bpy.context.scene.render.resolution_x = W 
    bpy.context.scene.render.resolution_y = H 

def import_file(obj_path):
    cube = bpy.data.objects['Cube']
    cube.select_set(True)
    bpy.ops.object.delete()
    lig = bpy.data.objects['Light']
    lig.select_set(True)
    bpy.ops.object.delete()
    rot_mat = np.eye(3)
    hom_coords = np.array([[0., 0., 0., 1.]]).reshape(1, 4)
    obj_location = np.zeros((1,3))
    obj_pose = np.concatenate((rot_mat, obj_location.reshape(3,1)), axis=-1)
    obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)
    import_mesh(obj_path, scale=1, object_world_matrix=obj_pose)
    obj = bpy.context.selected_objects[0]
    maxDim = max(obj.dimensions)
    obj.dimensions = obj.dimensions / maxDim

def clean_make_output_dirs(output_dir):
    img_dir = os.path.join(output_dir, 'rgb')
    pose_dir = os.path.join(output_dir, 'pose')
    seg_dir = os.path.join(output_dir, 'seg')
    uv_dir = os.path.join(output_dir, 'uv')
    normal_dir = os.path.join(output_dir, 'normal')
    depth_dir = os.path.join(output_dir, 'depth')
    seg_print_dir = os.path.join(output_dir, 'seg_colored')
    diffusion_dir = os.path.join(output_dir, 'diffusion')
    canny_dir = os.path.join(output_dir, 'canny')

    if os.path.exists(img_dir):  
        shutil.rmtree(img_dir)  
        os.makedirs(img_dir)
    else:
        os.makedirs(img_dir)


    if os.path.exists(seg_print_dir):  
        shutil.rmtree(seg_print_dir)  
        os.makedirs(seg_print_dir)
    else:
        os.makedirs(seg_print_dir)

    if os.path.exists(depth_dir):  
        shutil.rmtree(depth_dir)  
        os.makedirs(depth_dir)
    else:
        os.makedirs(depth_dir)

    if os.path.exists(diffusion_dir):  
        shutil.rmtree(diffusion_dir)  
        os.makedirs(diffusion_dir)
    else:
        os.makedirs(diffusion_dir)

    if os.path.exists(canny_dir):  
        shutil.rmtree(canny_dir)  
        os.makedirs(canny_dir)
    else:
        os.makedirs(canny_dir)
        
    if os.path.exists(seg_dir):  
        shutil.rmtree(seg_dir)  
        os.makedirs(seg_dir)
    else:
        os.makedirs(seg_dir)


    util.cond_mkdir(img_dir)
    util.cond_mkdir(pose_dir)
    util.cond_mkdir(seg_dir) 
    util.cond_mkdir(uv_dir)

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--obj_path', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--radius', type=float, default=1.8, help='Camera azimuth.')
parser.add_argument('--ren_num', type=int, default=10, help='Rendering frames number.')
parser.add_argument('--res_square', action='store_true', help='Padding CAD rendering into square for augmentation.')
argv = sys.argv
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

obj_path = args.obj_path
output_dir = args.output_dir
print('args.obj:' + obj_path)
print(output_dir)
render_num = args.ren_num
res_square = args.res_square

# set GPU render
set_gpu_renderer()

# set out size.
set_resolution()

# load obj files.
import_file(obj_path)

bpy.ops.transform.rotate(value=math.pi/2, orient_axis='X')

# set output nodes
bpy.context.scene.view_layers['ViewLayer'].use_pass_material_index = True
bpy.context.scene.view_layers['ViewLayer'].use_pass_uv = True
bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = True
bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True
mat_list = get_materials_names()

# set_trace()
for idx, mat in enumerate(mat_list):
    if mat == 'Dots Stroke' or mat == 'Material':
        bpy.data.materials[mat].pass_index = 0
    else:
        bpy.data.materials[mat].pass_index = int(mat.split('_')[1])+1
    
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
for n in tree.nodes:
    tree.nodes.remove(n)
# pass materials and material idx.
render_layers = tree.nodes.new('CompositorNodeRLayers')
alpha_node = tree.nodes.new(type="CompositorNodeAlphaOver")
alpha_node.premul = 1
output_node_img = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_img.format.color_mode = 'RGBA'
output_node_img.format.file_format = 'PNG'
links.new(render_layers.outputs['Image'], alpha_node.inputs[2])
links.new(alpha_node.outputs['Image'], output_node_img.inputs[0])
math_node = tree.nodes.new(type='CompositorNodeMath')
math_node.operation = 'DIVIDE'
math_node.inputs[1].default_value = 255.0
output_node_seg = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_seg.format.color_mode = 'BW'
links.new(render_layers.outputs['IndexMA'], math_node.inputs[0])
links.new(math_node.outputs['Value'], output_node_seg.inputs[0])

output_node_uv = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_uv.format.color_mode = 'RGB'
output_node_uv.format.file_format = 'OPEN_EXR'
links.new(render_layers.outputs['UV'], output_node_uv.inputs[0])

output_node_normal = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_normal.format.color_mode = 'RGB'
output_node_normal.format.file_format = 'OPEN_EXR'

output_node_depth = tree.nodes.new(type='CompositorNodeOutputFile')
output_node_depth.format.color_mode = 'BW'
output_node_depth.format.file_format = 'OPEN_EXR'
add_nodes = []
divide_nodes = []
for i in range(3):
    add_node = tree.nodes.new(type='CompositorNodeMath')
    # Set the operation to 'ADD', then divide the result by 2
    add_node.operation = 'ADD'
    add_node.inputs[1].default_value = 1  # The value to be added
    add_nodes.append(add_node)
    # To complete the operation (value + 1) / 2, we need another math node for division
    divide_node = tree.nodes.new(type='CompositorNodeMath')
    divide_node.location = add_node.location.x + 200, math_node.location.y  # Positioning next to the add node
    divide_node.operation = 'DIVIDE'
    divide_node.inputs[1].default_value = 2  # Dividing by 2
    divide_nodes.append(divide_node)
# Create a Separate RGB node
separate_rgb = tree.nodes.new(type='CompositorNodeSepRGBA')
separate_rgb.location = 200, 0  # Adjust the location as needed
links.new(render_layers.outputs['Normal'], separate_rgb.inputs[0])
combine_rgb = tree.nodes.new(type='CompositorNodeCombRGBA')
for i, channel in enumerate(['R', 'G', 'B']):
    links.new(separate_rgb.outputs[channel], add_nodes[i].inputs[0])
    links.new(add_nodes[i].outputs[0], divide_nodes[i].inputs[0])
    links.new(divide_nodes[i].outputs[0], combine_rgb.inputs[channel])
links.new(combine_rgb.outputs[0], output_node_normal.inputs[0])
# print(render_layers.outputs.keys())
links.new(render_layers.outputs['Depth'], output_node_depth.inputs[0])

# set uniform lights
set_uniform_lights()

# set camera.
cam_locations = get_archimedean_spiral(args.radius, render_num)
# fit the camera height in the kitti360 dataset
cam_locations[:, 1] /= 2.5
cam = bpy.data.objects['Camera']
obj_location = np.zeros((1,3))
cv_poses = util.look_at(cam_locations, obj_location)
blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
K = util.get_calibration_matrix_K_from_blender(cam.data)
fov = cam.data.angle # Y field of view angle

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'fov.txt'),'w') as fov_files:
    fov_files.write('%f\n'%fov)

# from ipdb import set_trace
# set_trace()

K = np.asarray(K)
with open(os.path.join(output_dir, 'cam_K.txt'),'w') as f:
    f.write("%f %f %f\n" % (K[0, 0], K[0, 1], K[0, 2]))
    f.write("%f %f %f\n" % (K[1, 0], K[1, 1], K[1, 2]))
    f.write("%f %f %f\n" % (K[2, 0], K[2, 1], K[2, 2]))

# clear and make output directories
clean_make_output_dirs(output_dir)

# set node output directories
img_dir = os.path.join(output_dir, 'rgb')
pose_dir = os.path.join(output_dir, 'pose')
seg_dir = os.path.join(output_dir, 'seg')
uv_dir = os.path.join(output_dir, 'uv')
normal_dir = os.path.join(output_dir, 'normal')
depth_dir = os.path.join(output_dir, 'depth')
output_node_img.base_path = img_dir
output_node_seg.base_path = seg_dir
output_node_uv.base_path = uv_dir
output_node_normal.base_path = normal_dir
output_node_depth.base_path = depth_dir

blender_cam2world_matrices = blender_poses
camera = cam
for i in range(len(blender_cam2world_matrices)):
        camera.matrix_world = blender_cam2world_matrices[i]
        output_node_img.file_slots[0].path = '{:06d}_'.format(i)
        output_node_seg.file_slots[0].path = '{:06d}_'.format(i)
        output_node_uv.file_slots[0].path = '{:06d}_'.format(i)
        output_node_normal.file_slots[0].path = '{:06d}_'.format(i)
        output_node_depth.file_slots[0].path = '{:06d}_'.format(i)
        # Render the color image
        bpy.ops.render.render(write_still=True)
        # Write out camera pose
        RT = util.get_world2cam_from_blender_cam(cam)
        cam2world = RT.inverted()
        with open(os.path.join(pose_dir, '%06d.txt'%i),'w') as pose_file:
            matrix_flat = []
            for j in range(4):
                for k in range(4):
                    matrix_flat.append(cam2world[j][k])
            pose_file.write(' '.join(map(str, matrix_flat)) + '\n')
