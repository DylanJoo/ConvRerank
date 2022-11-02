python3 xla_spawn.py \
  --num_cores 8 \
  train_dev.py \
  --resume_from_checkpoint ./checkpoints/colbertv2.0 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./checkpoints/licqe-ibn.v0 \
  --train_file ../convir_data/canard_convir.train.quadruples.cqe.v0.jsonl \
  --max_c_seq_length 32 \
  --max_u_seq_length 32 \
  --max_p_seq_length 150 \
  --colbert_type 'colbert-inbatch' \
  --dim 128 \
  --remove_unused_columns false \
  --per_device_train_batch_size 12 \
  --learning_rate 7e-6 \
  --max_steps 15000 \
  --save_steps 5000 \
  --do_train
