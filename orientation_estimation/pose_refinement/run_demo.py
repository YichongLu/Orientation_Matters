# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

def mesh_normalization(mesh):
    
    bbox_min = mesh.bounds[0]  # 最小点
    bbox_max = mesh.bounds[1]  # 最大点
    center = (bbox_min + bbox_max) / 2  # 模型中心
    scale = np.max(bbox_max - bbox_min)  # 最大边长
    scale = scale
    return scale

def align_mesh_to_metric_scale(mesh, test_scene_dir):
    # 读取测试场景的深度图
    depth_file = os.path.join(test_scene_dir, 'depth', test_scene_dir.split('/')[-1] + '.png')
    depth = cv2.imread(depth_file, -1)
    depth = depth.astype(np.int16) / 1000.0
    mask = cv2.imread(os.path.join(test_scene_dir, 'masks', test_scene_dir.split('/')[-1] + '.png'), -1)[:,:,0]
    mask = mask.astype(bool)
    depth[~mask] = 0
    K = np.loadtxt(os.path.join(test_scene_dir, 'cam_K.txt'))
    rgb = cv2.imread(os.path.join(test_scene_dir, 'rgb', test_scene_dir.split('/')[-1] + '.png'))
    
    # 将深度图转换为xyz坐标
    # xyz_map = depth2xyzmap(depth, K)
    
    # Convert depth map to point cloud using K
    points_3d = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u]
            if z > 0.001:  # Check if depth is valid
                x = (u - K[0, 2]) * z / K[0, 0]
                y = (v - K[1, 2]) * z / K[1, 1]
                points_3d[v, u] = (x, y, z)
    
    # 把xyz坐标转换为open3d点云
    # valid = xyz_map[..., 2] > 0
    # pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
    # o3d.io.write_point_cloud(os.path.join("./test/", 'aligned_pointcloud.ply'), pcd)
    valid = points_3d[..., 2] > 0
    pcd = toOpen3dCloud(points_3d[valid], rgb[valid])
    o3d.io.write_point_cloud(os.path.join("./test/", 'aligned_pointcloud.ply'), pcd)
    
    # 把trimesh模型转换为open3d模型
    mesh_open3d = trimesh_to_open3d(mesh)
    o3d.io.write_triangle_mesh(os.path.join("./test/", 'aligned_mesh.ply'), mesh_open3d)
    
    # Step 1: 从Mesh中采样点云（与输入点云点数一致）
    mesh_pcd = mesh_open3d.sample_points_uniformly(number_of_points=len(pcd.points))
    
    # Step 2: 获取点云和Mesh的坐标数组
    points_pcd = np.asarray(pcd.points)
    points_mesh = np.asarray(mesh_pcd.points)
    
    # Step 3: 中心化数据（减去均值）
    mean_pcd = np.mean(points_pcd, axis=0)
    mean_mesh = np.mean(points_mesh, axis=0)
    
    centered_pcd = points_pcd - mean_pcd
    centered_mesh = points_mesh - mean_mesh
    
    # Step 4: 计算PCA（协方差矩阵的特征分解）
    cov_pcd = np.cov(centered_pcd, rowvar=False)
    cov_mesh = np.cov(centered_mesh, rowvar=False)
    
    eigvals_pcd, eigvecs_pcd = np.linalg.eig(cov_pcd)
    eigvals_mesh, eigvecs_mesh = np.linalg.eig(cov_mesh)
    
    # Step 5: 计算缩放因子（按Mesh的PCA尺度缩放点云）
    scale_factors = np.sqrt(eigvals_mesh) / np.sqrt(eigvals_pcd)
    scale_factor = np.mean(scale_factors)
    mesh.apply_scale(1/scale_factor)
    
    return mesh
    
    
def trimesh_to_open3d(mesh):
    """
    Convert a trimesh mesh to an open3d triangle mesh.
    
    Parameters:
    - mesh: trimesh.Trimesh object
    
    Returns:
    - open3d.geometry.TriangleMesh object
    """
    vertices = mesh.vertices
    triangles = mesh.faces
    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return triangle_mesh
    
    

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, required=True)
  parser.add_argument('--root_dir', type=str, required=True)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--mesh_rescale_mode', type=str, default='normalization')
  parser.add_argument('--bg_color', type=str, default='black')
  parser.add_argument('--object_level', type=bool, default=True)
  parser.add_argument('--rgb_only', type=bool, default=True)
  parser.add_argument('--crop_ratio', type=float, default=1.8)
  parser.add_argument('--ckpt_root_dir', type=str, default='./weights')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  
  test_scene_dir = os.path.join(args.root_dir, "foundationpose_inputs")
  output_dir = os.path.join(args.root_dir, "foundationpose_outputs")
  
  mesh = trimesh.load(args.mesh_file, force='mesh')
  
  if args.mesh_rescale_mode == 'normalization':
    scale = mesh_normalization(mesh)
    mesh.apply_scale(1/scale)
  elif args.mesh_rescale_mode == 'metric_scale':
    mesh = align_mesh_to_metric_scale(mesh, test_scene_dir)
  elif args.mesh_rescale_mode == 'no_rescale':
    mesh = mesh
  else:
    raise ValueError(f'Invalid mesh rescale mode: {args.mesh_rescale_mode}')
  
  
  debug = args.debug
  os.system(f'rm -rf {output_dir}/* && mkdir -p {output_dir}/track_vis {output_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor(ckpt_root_dir=os.path.join(args.ckpt_root_dir, 'pose_refinement', 'ckpts'))
  refiner = PoseRefinePredictor(ckpt_root_dir=os.path.join(args.ckpt_root_dir, 'pose_refinement', 'ckpts'))
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=output_dir, debug=debug, glctx=glctx, bg_color=args.bg_color, object_level=args.object_level, rgb_only=args.rgb_only, crop_ratio=args.crop_ratio)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        xyz_map = depth2xyzmap(depth, reader.K) # calculate the xy coordinates of the points in the world coordinate system
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid]) # create a point cloud from the xy coordinates and the color of the points
        o3d.io.write_point_cloud(f'{output_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{output_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{output_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    
    np.savetxt(f'{output_dir}/refiner/K.txt', reader.K)

    if debug>=1:
      # center_pose = pose@np.linalg.inv(to_origin)
      center_pose = pose.reshape(4,4)
      # vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)


    if debug>=2:
      os.makedirs(f'{output_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{output_dir}/track_vis/{reader.id_strs[i]}.png', vis)

