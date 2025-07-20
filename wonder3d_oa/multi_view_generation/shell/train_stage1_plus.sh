
GPU_CONFIG_FILE=${1}
TRAINING_CONFIG_FILE=${2}

accelerate launch --config_file ${GPU_CONFIG_FILE} train_mvdiffusion_image_plus.py --config ${TRAINING_CONFIG_FILE}