python3 tools/rewrite_cqr.py \
  --topics data/cast2019/cast2019.eval.jsonl \
  --output data/cast2019/cast2019.eval.rewrite.tsv \
  --beam_size 10 \
  --model_name castorini/t5-base-canard
