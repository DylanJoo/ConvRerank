mkdir -p data/canard4ir/

# Setting 1: view1 - view0
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
  --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.txt \
  --window_size 3 \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \

# Setting 2: view0 - viewx
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.jsonl \
#   --collections /tmp2/jhju/datasets/cast2020/ \
#   --run0 runs/cast20.canard.train.viewx.monot5.top1000.trec \
#   --run1 runs/cast20.canard.train.view0.monot5.top1000.trec \
#   --output data/canard4ir/canard4ir.train.convrerank.setting2.txt \
#   --window_size 3 \
#   --topk_pool 200 \
#   --topk_positive 3 \
#   --n 30 \
