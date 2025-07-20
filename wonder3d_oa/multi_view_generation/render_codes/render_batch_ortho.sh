 python distributed.py \
	--num_gpus 6 --gpu_list 2 3 4 5 6 7 --mode render_ortho    \
	--workers_per_gpu 10 --view_idx 0 \
	--start_i 0 --end_i 100000 --ortho_scale 1.35 \
	--input_models_path ../data_lists/lvis_uids_filter_by_vertex.json  \
	--objaverse_root /nas/datasets/Objaverse_lvis/glbs \
	--save_folder /data2/yclu/Pose_matching/wonder3d_training_data/objaverse_rendering \
	--blender_install_path /data2/yclu/blender-3.3.0-linux-x64/ \
	--random_pose
