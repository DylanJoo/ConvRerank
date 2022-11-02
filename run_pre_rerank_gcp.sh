# baseline reranking model castorini/monot5-base-msmarco (100K)
python3 passage_reranking.py \
  --run gs://cnclab/cast/runs/cast20.canard.train.answer+rewrite.spr.top1000.trec \
  --topic data/canard.train.jsonl \
  --collection gs://cnclab/cast/collections/cast2020_psg/ \
  --topk 1000 \
  --output_trec runs/cast20.canard.train.answer+rewrite.rerank.top1000.trec \
  --model_name_or_path castorini/monot5-base-msmarco \
  --batch_size 4 \
  --max_length 512 \
  --gpu 1 \
  --prefix monot5-base 

# python3 passage_reranking.py \
#   --run gs://cnclab/cast/runs/cast20.canard.train.rewrite.spr.top1000.trec \
#   --topic data/canard.train.jsonl \
#   --collection gs://cnclab/cast/collections/cast2020_psg \
#   --topk 1000 \
#   --output_trec gs://cnclab/cast/runs/cast20.canard.train.rewrite.rerank.top1000.trec \
#   --model_name_or_path castorini/monot5-base-msmarco \
#   --batch_size 4 \
#   --max_length 512 \
#   --gpu 0 \
#   --prefix monot5-base
