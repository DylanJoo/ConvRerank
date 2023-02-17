mkdir -p data/canard4ir/
topk=40
# Main-B: hard negative from view0
python3 tools/analyze_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view0.monot5.top1000.trec \
  --run1 runs/rerank_step1/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/analyze_t40.txt \
  --topk_pool 200 \
  --topk_positive 40 \
  --n 40

