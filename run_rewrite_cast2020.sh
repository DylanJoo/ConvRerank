python3 tools/rewrite_cqr.py \
  --topics data/cast2020/cast2020.eval.jsonl \
  --output data/cast2020/cast2020.eval.rewrite.tsv \
  --beam_size 10 \
  --model_name castorini/t5-base-canard
