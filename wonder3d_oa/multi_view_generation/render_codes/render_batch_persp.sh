 python distributed.py \
	--num_gpus 4 --gpu_list 0 1 2 3 --mode render_persp    \
	--workers_per_gpu 10 --view_idx 0 \
	--input_models_path /home/yclu/Wonder3D-main/data_lists/cano_objaverse_uids_v2.json \
	--save_folder /data2/yclu/wonder3d/wonder3d_training_data/cano_objaverse_lvis_rendering_v2 \
	--dataset_root /data2/yclu/Pose_matching/Version_2/cano_objaverse_lvis_v2/ \
	--blender_install_path /data2/yclu/blender-3.3.0-linux-x64/ \
