
import argparse
import os
import multiprocessing
import subprocess
from dataclasses import dataclass
from typing import Optional

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        
        gpu_id = gpu
        
        mesh_file = item[0]
        test_scene_dir = item[1]
        debug_dir = item[2]
        os.makedirs(debug_dir, exist_ok=True)
        command1 = f"CUDA_VISIBLE_DEVICES={gpu_id} python FoundationPose/run_demo.py --mesh_file {mesh_file} --test_scene_dir {test_scene_dir} --debug_dir {debug_dir}"
    
        print(command1)
        subprocess.run(command1, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()





if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--inputs_dir', type=str, default="/data2/yclu/foundationpose/inputs/gso_30")
  parser.add_argument('--mesh_dir', type=str, default="/data2/yclu/wonder3d/evaluation/outputs/gso_30/wonder3d_exp10J_ckpt_75000_epochs/3d_results/lgm")
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--debug_dir', type=str, default="/data2/yclu/foundationpose/outputs/gso_30_lgm")
  parser.add_argument('--gpu_id', type=int, default=0)
  parser.add_argument('--method', type=str, default="trellis")
  args = parser.parse_args()
  
  inputs_dir = args.inputs_dir


  queue = multiprocessing.JoinableQueue()
  count = multiprocessing.Value("i", 0)

  num_gpus = 1
  workers_per_gpu = 3
  gpu_list = [args.gpu_id]

  # Start worker processes on each of the GPUs
  for gpu_i in range(num_gpus):
      for worker_i in range(workers_per_gpu):
          worker_i = gpu_i *workers_per_gpu + worker_i
          process = multiprocessing.Process(
              target=worker, args=(queue, count, gpu_list[gpu_i])
          )
          process.daemon = True
          process.start()

  mesh_files = []
  test_scene_dirs = []
  debug_dirs = []
  
  for object_name in os.listdir(inputs_dir):
    print(object_name)
    if args.method == "trellis":
      mesh_file = os.path.join(args.mesh_dir, object_name, 'sample.glb')
    elif args.method == "wonder3d":
      mesh_file = os.path.join(args.mesh_dir, object_name+'_rotated.glb')
    test_scene_dir = os.path.join(inputs_dir, object_name)
    debug_dir = os.path.join(args.debug_dir, object_name)
    
    
    mesh_files.append(mesh_file)
    test_scene_dirs.append(test_scene_dir)
    debug_dirs.append(debug_dir)
      
  for mesh_file, test_scene_dir, debug_dir in zip(mesh_files, test_scene_dirs, debug_dirs):
      queue.put([mesh_file, test_scene_dir, debug_dir])
      
  queue.join()

  for i in range(num_gpus * workers_per_gpu):
      queue.put(None)


