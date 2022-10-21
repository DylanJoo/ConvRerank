python3 tools/construct_convir_dataset.py \
  --topic data/canard.train.jsonl \
  --run_teacher runs/cast20.canard.train.answer+rewrite.spr.top1000.trec \
  --run_student runs/cast20.canard.train.rewrite.spr.top1000.trec  \
  --output data/convir/convir.train.convrerank.jsonl \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \
  --collections /tmp2/trec/cast/2020/cast2020_psg/
