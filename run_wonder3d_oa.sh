python wonder3d_oa/multi_view_generation/test_mvdiffusion_seq.py --config wonder3d_oa/multi_view_generation/configs/test_config.yaml validation_dataset.root_dir=$1 validation_dataset.filepaths=[$2] save_dir=$3/mv_results pretrained_model_name_or_path=$4/multi_view_generation/pretrained_ckpts lora_output_dir=$4/multi_view_generation/lora_ckpt load_lora=true

python wonder3d_oa/3d_lifting/mv2mesh.py big --resume $4/3d_lifting/ckpts/model_fp16.safetensors --save_dir $3/wonder3d_oa_results --input_dir $3/mv_results/cropsize-192-cfg3.0

python wonder3d_oa/3d_lifting/convert.py big --input_path $3/wonder3d_oa_results/sample.ply

python wonder3d_oa/3d_lifting/rotate_lgm_glb_w_trimesh.py --model_path $3/wonder3d_oa_results/sample.glb
