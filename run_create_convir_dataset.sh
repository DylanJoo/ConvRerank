mkdir -p data/canard4ir/


# Main-A: hard sampling from view disagreement
for k in 10 20 30 40 50 100;do
    python3 tools/construct_convir_dataset.py \
      --topic data/canard/train.jsonl \
      --collections /tmp2/jhju/datasets/cast2020/ \
      --run0 runs/rerank_step1/cast20.canard.train.view0.monot5.top1000.trec \
      --run1 runs/rerank_step1/cast20.canard.train.view1.monot5.top1000.trec \
      --output data/canard4ir/canard4ir.train.convrerank.t${k}.txt \
      --topk_pool 200 \
      --topk_positive $k \
      --n $k &
done

topk=40
# Main-B: hard negative from view0
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view0.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.singleview0.txt \
  --topk_pool 200 \
  --topk_positive $topk \
  --n $topk \
  --singleview 

# Main-B: hard negative from view1
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.singleview1.txt \
  --topk_pool 200 \
  --topk_positive $topk \
  --n $topk \
  --singleview

# Main-B: hard negative from view disagreement [top20, reverview]
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections /tmp2/jhju/datasets/cast2020/ \
  --run0 runs/rerank_step1/cast20.canard.train.view1.monot5.top1000.trec \
  --run1 runs/rerank_step1/cast20.canard.train.view0.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.reverseview.txt \
  --topk_pool 200 \
  --topk_positive $topk \
  --n $topk
