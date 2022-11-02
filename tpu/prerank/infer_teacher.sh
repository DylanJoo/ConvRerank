HOME="/home/g110753140"
PROJECT_DIR="${HOME}/convrerank/torank"
T5X_DIR="${HOME}/t5x"
INFER_OUTPUT_DIR="gs://cnclab/cast/infer/teacher/"  # directory to write infer output
CHECKPOINT_PATH="gs://castorini/monot5/experiments/base/model.ckpt-1100000" # you can also use mtf 
MODE="score"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/infer.py \
	--gin_search_paths=${PROJECT_DIR} \
	--gin_file="${PROJECT_DIR}/torank_teacher.gin" \
        --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
        --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
	--gin.infer.mode=\"${MODE}\" \
        --gin.USE_CACHED_TASKS="False"
