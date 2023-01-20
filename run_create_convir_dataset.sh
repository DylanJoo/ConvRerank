mkdir -p data/canard4ir/


# Main-A: hard negative from view disagreement [top20]
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.jsonl \
#   --collections /tmp2/jhju/datasets/cast2020/ \
#   --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
#   --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
#   --output data/canard4ir/canard4ir.train.convrerank.t20.txt \
#   --topk_pool 200 \
#   --topk_positive 20 \
#   --n 20 
#

# Main-B: hard negative from view0
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view0.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.singleview0.txt \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 20 \
  --singleview 
# Main-B: hard negative from view1
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.singleview1.txt \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 20 \
  --singleview
# Main-B: hard negative from view disagreement [top20, reverview]
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view1.monot5.top1000.trec \
  --run1 runs/rerank_step1/cast20.canard.train.view0.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.reverseview.txt \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 20

# # Ablation-A: hard negative from view disagreement [top3]
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.jsonl \
#   --collections /tmp2/jhju/datasets/cast2020/ \
#   --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
#   --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
#   --output data/canard4ir/canard4ir.train.convrerank.t3.txt \
#   --topk_pool 200 \
#   --topk_positive 3 \
#   --n 20 
# # Ablation-A: hard negative from view disagreement [top10]
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.jsonl \
#   --collections /tmp2/jhju/datasets/cast2020/ \
#   --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
#   --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
#   --output data/canard4ir/canard4ir.train.convrerank.t10.txt \
#   --topk_pool 200 \
#   --topk_positive 10 \
#   --n 20
# Ablation-A: hard negative from view disagreement [top30]
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.jsonl \
#   --collections /tmp2/jhju/datasets/cast2020/ \
#   --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
#   --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
#   --output data/canard4ir/canard4ir.train.convrerank.t30.txt \
#   --topk_pool 200 \
#   --topk_positive 30 \
#   --n 20

# Abalation-B: Set-based positive mining [j20]
# python3 tools/construct_convir_dataset_dev.py \
#   --topic data/canard/train.jsonl \
#   --collections /tmp2/jhju/datasets/cast2020/ \
#   --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
#   --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
#   --output data/canard4ir/canard4ir.train.convrerank.j20.txt \
#   --topk_pool 200 \
#   --topk_sample 20

