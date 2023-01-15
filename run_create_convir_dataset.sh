mkdir -p data/canard4ir/

# singleview: 
# ablation 1: hard negative from view0
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.singleview0.txt \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \
  --singleview

# hard negative from view1
# ablation 2: hard negative from view1
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.singleview1.txt \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \
  --singleview

# dualview
# setting 1: hard negative from view disagreement
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
  --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.txt \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \
