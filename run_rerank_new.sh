candidate=t40
# Main-A: monot5 rerank & Convrerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.conv.rerank.$candidate.txt.probs \
    --reranked runs/cast2019/new.trec \
    --topk 200 \
    --prefix conv-monot5

python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2020/cast2020.eval.cqe.conv.rerank.$candidate.txt.probs \
    --reranked runs/cast2020/new.trec \
    --topk 200 \
    --prefix conv-monot5

python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.rerank.txt.probs \
    --reranked runs/cast2019/new2.trec \
    --topk 200 \
    --prefix monot5

python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2020/cast2020.eval.cqe.rerank.txt.probs \
    --reranked runs/cast2020/new2.trec \
    --topk 200 \
    --prefix monot5

echo 2019
./trec_eval-9.0.7/trec_eval -c \
    -m ndcg_cut.3,5,10,1000 \
    data/cast2019/2019qrels.txt runs/cast2019/new.trec 
./trec_eval-9.0.7/trec_eval -c \
    -m ndcg_cut.3,5,10,1000 \
    data/cast2019/2019qrels.txt runs/cast2019/new2.trec 

echo 2020
./trec_eval-9.0.7/trec_eval \
    -c -m ndcg_cut.3,5,10,1000 \
    data/cast2020/2020qrels.txt runs/cast2020/new.trec 
./trec_eval-9.0.7/trec_eval \
    -c -m ndcg_cut.3,5,10,1000 \
    data/cast2020/2020qrels.txt runs/cast2020/new2.trec 
