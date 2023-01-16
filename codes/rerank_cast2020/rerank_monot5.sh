YOUR_TPU=node-1
PRJ_NAME=convrerank
TPU_PLACE=us-central1-f
MODEL_DIR=castorini/monot5/experiments/base
MODEL_SIZE=base
CKPT=1100000
INPUT_FILE=convrerank/rerank_cast2020/input/cast2020.eval.cqe.rerank.txt
TARGET_FILE=convrerank/true.txt
OUTPUT_FILE=convrerank/rerank_cast2020/output/cast2020.eval.cqe.rerank.txt


echo "Input: ${INPUT_FILE}\t Output: ${OUTPUT_FILE}" > $YOUR_TPU.log
t5_mesh_transformer \
  --tpu="${YOUR_TPU}" \
  --gcp_project="${PRJ_NAME}" \
  --tpu_zone="${TPU_PLACE}" \
  --model_dir="gs://${MODEL_DIR}" \
  --gin_file="gs://t5-data/pretrained_models/${MODEL_SIZE}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="score_from_file.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="infer_checkpoint_step = ${CKPT}" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 4}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="inputs_filename = 'gs://${INPUT_FILE}'" \
  --gin_param="targets_filename = 'gs://${TARGET_FILE}'" \
  --gin_param="scores_filename = 'gs://${OUTPUT_FILE}'" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" 

gsutil rm gs://$OUTPUT_FILE.targets
gsutil rm gs://$OUTPUT_FILE.length

t5_mesh_transformer \
  --tpu="${YOUR_TPU}" \
  --gcp_project="${PRJ_NAME}" \
  --tpu_zone="${TPU_PLACE}" \
  --model_dir="gs://${MODEL_DIR}" \
  --gin_file="gs://t5-data/pretrained_models/${MODEL_SIZE}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="score_from_file.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="infer_checkpoint_step = ${CKPT}" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 4}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="inputs_filename = 'gs://${INPUT_FILE/cqe/t5-cqe}'" \
  --gin_param="targets_filename = 'gs://${TARGET_FILE}'" \
  --gin_param="scores_filename = 'gs://${OUTPUT_FILE/cqe/t5-cqe}'" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" 

gsutil rm gs://${OUTPUT_FILE/cqe/t5-cqe}.targets
gsutil rm gs://${OUTPUT_FILE/cqe/t5-cqe}.lengths
