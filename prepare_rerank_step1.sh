# View0 ranked list
python3 tools/convert_runs_to_monot5.py \
  --run runs/cast20.canard.train.view0.bm25.top1000.trec \
  --topic data/canard/train.jsonl \
  --collection /tmp2/jhju/datasets/cast2020/ \
  --output monot5/cast20.canard.train.view0.rerank.txt

# View1 ranked list
python3 tools/convert_runs_to_monot5.py \
  --run runs/cast20.canard.train.view1.bm25.top1000.trec \
  --topic data/canard/train.jsonl \
  --collection /tmp2/jhju/datasets/cast2020/ \
  --output monot5/cast20.canard.train.view1.rerank.txt

# Viewx ranked list
python3 tools/convert_runs_to_monot5.py \
  --run runs/cast20.canard.train.viewx.bm25.top1000.trec \
  --topic data/canard/train.jsonl \
  --collection /tmp2/jhju/datasets/cast2020/ \
  --output monot5/cast20.canard.train.viewx.rerank.txt

