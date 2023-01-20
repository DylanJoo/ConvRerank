mkdir -p runs/cast2020
candidates=$1

# rerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.rerank.trec \
    --topk 1000 \
    --prefix monot5 
# conv rerank (zs)
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.zs.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.zs.rerank.trec \
    --topk 1000 \
    --prefix monot5-zs
# conv rerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.conv.rerank.t20.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.t20.trec \
    --topk 1000 \
    --prefix conv-monot5 

# abalation-A: [Select topK]
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.conv.rerank.t3.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.t3.trec \
    --topk 1000 \
    --prefix conv-monot5 
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.conv.rerank.t10.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.t10.trec \
    --topk 1000 \
    --prefix conv-monot5 
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.conv.rerank.t30.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.t30.trec \
    --topk 1000 \
    --prefix conv-monot5 
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/cast2020.eval.cqe.conv.rerank.j20.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.j20.trec \
    --topk 1000 \
    --prefix conv-monot5 

echo 'Run, R@100, R@1000, nDCG@3, nDCG@5, nDCG@10'
for run in runs/cast2020/*;do
    echo -n ${run##*cast2020.}','
    ./trec_eval-9.0.7/trec_eval \
        -m recall.100,1000 -m ndcg_cut.3,5,10 \
        data/cast2020/2020qrels.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done
