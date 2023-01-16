YOUR_TPU=node-3
PRJ_NAME=convrerank            
TPU_PLACE=europe-west4-a
MODEL_DIR_INIT=castorini/monot5/experiments/large
MODEL_SIZE=large
CKPT_INIT=1100700
CKPT_FINAL=1120700
TRAIN_FILE=convrerank/dataset/canard4ir.train.convrerank.txt
MODEL_DIR=convrerank/checkpoints/monot5-large-canard4ir

t5_mesh_transformer  \
  --tpu="${YOUR_TPU}" \
  --gcp_project="${PRJ_NAME}" \
  --tpu_zone="${TPU_PLACE}" \
  --model_dir="gs://${MODEL_DIR}" \
  --gin_param="init_checkpoint = 'gs://${MODEL_DIR_INIT}/model.ckpt-${CKPT_INIT}'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/${MODEL_SIZE}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://${TRAIN_FILE}'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = ${CKPT_FINAL}" \
  --gin_param="run.save_checkpoints_steps = 10000" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 4}" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)"
