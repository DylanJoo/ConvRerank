# Although there are many ways to inference (for 'pre' re-rank)
# 
# As it can be inferenced and reranked with 'passage_ranking.py',
# Inferencing on TPU will be much more efficient.
# Therefore, prepare the results and store in gbucket for TPU inferencing

mkdir data/torank/

# teacher results
python3 tools/convert_runs_to_monot5.py \
  --run runs/cast20.canard.train.answer+rewrite.spr.top1000.trec \
  --topic data/canard.train.jsonl \
  --collection /tmp2/trec/cast/2020/cast2020_psg \
  --output data/torank/cast20.canard.train.answer+rewrite.rerank.top1000.trec \
  --batch_size 1000

# student results
python3 tools/convert_runs_to_monot5.py \
  --run runs/cast20.canard.train.rewrite.spr.top1000.trec \
  --topic data/canard.train.jsonl \
  --collection /tmp2/trec/cast/2020/cast2020_psg \
  --output data/torank/cast20.canard.train.rewrite.rerank.top1000.trec \
  --batch_size 1000

