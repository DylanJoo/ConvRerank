mkdir -p runs/test

# Main-A: monot5 rerank & Convrerank
for k in 10 20 40 50 100 300 500 1000;do
    python3 tools/rerank_runs.py \
        --baseline runs/cast2019/cast2019.eval.hqe.trec \
        --scores monot5-probs/rerank_cast2019/ablation_firststage/cast2019.eval.hqe.rerank.txt.probs \
        --reranked runs/test/monot5-$k.trec \
        --topk $k \
        --prefix monot5 

    python3 tools/rerank_runs.py \
        --baseline runs/cast2019/cast2019.eval.hqe.trec \
        --scores monot5-probs/rerank_cast2019/ablation_firststage/cast2019.eval.hqe.conv.rerank.txt.probs \
        --reranked runs/test/convrerank-$k.trec \
        --topk $k \
        --prefix conv-monot5 
done

echo 'Run, Recall@100'
for run in runs/test/*.trec;do
    echo -n $run','
    ./trec_eval-9.0.7/trec_eval -c \
        -c -m ndcg_cut.3 \
        data/cast2019/2019qrels.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done

