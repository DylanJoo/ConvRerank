python3 tools/rerank_runs.py \
    --baseline runs/cast20.canard.train.view0.bm25.top1000.trec \
    --scores monot5-probs/cast20.canard.train.view0.rerank.probs \
    --reranked runs/cast20.canard.train.view0.monot5.top1000.trec \
    --topk 1000 \
    --prefix monot5 

python3 tools/rerank_runs.py \
    --baseline runs/cast20.canard.train.view1.bm25.top1000.trec \
    --scores monot5-probs/cast20.canard.train.view1.rerank.probs \
    --reranked runs/cast20.canard.train.view1.monot5.top1000.trec \
    --topk 1000 \
    --prefix monot5 

python3 tools/rerank_runs.py \
    --baseline runs/cast20.canard.train.viewx.bm25.top1000.trec \
    --scores monot5-probs/cast20.canard.train.viewx.rerank.probs \
    --reranked runs/cast20.canard.train.viewx.monot5.top1000.trec \
    --topk 1000 \
    --prefix monot5 

