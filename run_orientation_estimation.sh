python orientation_estimation/data_preprocessing/scripts/inference.py --image_path $1 --output_dir $2 --ckpt_root_dir $4

python orientation_estimation/foundationpose_preprocess.py --image_path $1 --output_dir $2

python orientation_estimation/pose_refinement/run_demo.py --root_dir $2 --mesh_file $3 --ckpt_root_dir $4

python orientation_estimation/pose_selection/dinov2_pose.py --root_dir $2 --test_image $1 --ckpt_root_dir $4