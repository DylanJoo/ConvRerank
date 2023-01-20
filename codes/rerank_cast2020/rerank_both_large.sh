YOUR_TPU=node-3
PRJ_NAME=convrerank
TPU_PLACE=europe-west4-a
MODEL_SIZE=large

MODEL_DIR=convrerank/checkpoints/monot5-large-canard4ir
CKPT=1110800
INPUT_FILE=convrerank/rerank_cast2020/input/cast2020.eval.cqe.conv.rerank.txt
TARGET_FILE=convrerank/true.txt
OUTPUT_FILE=convrerank/rerank_cast2020/ablation_large/cast2020.eval.cqe.conv.rerank.txt

echo "Input: ${INPUT_FILE}\t Output: ${OUTPUT_FILE}" > $YOUR_TPU.log
t5_mesh_transformer \
  --tpu="${YOUR_TPU}" \
  --gcp_project="${PRJ_NAME}" \
  --tpu_zone="${TPU_PLACE}" \
  --model_dir="gs://${MODEL_DIR}" \
  --gin_file="gs://t5-data/pretrained_models/${MODEL_SIZE}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="score_from_file.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="infer_checkpoint_step = ${CKPT}" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 4}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="inputs_filename = 'gs://${INPUT_FILE}'" \
  --gin_param="targets_filename = 'gs://${TARGET_FILE}'" \
  --gin_param="scores_filename = 'gs://${OUTPUT_FILE}'" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" 

gsutil rm gs://${OUTPUT_FILE}.lengths
gsutil rm gs://${OUTPUT_FILE}.targets

MODEL_DIR=castorini/monot5/experiments/large
CKPT=1100700
INPUT_FILE=convrerank/rerank_cast2020/input/cast2020.eval.cqe.rerank.txt
TARGET_FILE=convrerank/true.txt
OUTPUT_FILE=convrerank/rerank_cast2020/ablation_large/cast2020.eval.cqe.rerank.txt
t5_mesh_transformer \
  --tpu="${YOUR_TPU}" \
  --gcp_project="${PRJ_NAME}" \
  --tpu_zone="${TPU_PLACE}" \
  --model_dir="gs://${MODEL_DIR}" \
  --gin_file="gs://t5-data/pretrained_models/${MODEL_SIZE}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="score_from_file.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="infer_checkpoint_step = ${CKPT}" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 4}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="inputs_filename = 'gs://${INPUT_FILE}'" \
  --gin_param="targets_filename = 'gs://${TARGET_FILE}'" \
  --gin_param="scores_filename = 'gs://${OUTPUT_FILE}'" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" 

gsutil rm gs://${OUTPUT_FILE}.lengths
gsutil rm gs://${OUTPUT_FILE}.targets