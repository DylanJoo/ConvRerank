candidates=cqe
# rerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.$candidates.trec \
    --scores monot5-probs/cast2020.eval.$candidates.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.$candidates.rerank.trec \
    --topk 1000 \
    --prefix monot5 

# conv rerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.$candidates.trec \
    --scores monot5-probs/cast2020.eval.$candidates.conv.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.$candidates.conv.rerank.trec \
    --topk 1000 \
    --prefix conv-monot5 

