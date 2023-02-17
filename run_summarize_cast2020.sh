python3 tools/summarize_t5.py \
  --topics data/cast2020/cast2020.eval.jsonl \
  --output data/cast2020/cast2020.eval.summary.tsv \
  --collection /tmp2/jhju/datasets/cast2020 \
  --seq_length 16 32 \
  --beam_size 5 
