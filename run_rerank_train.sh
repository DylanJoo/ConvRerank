mkdir -p runs/cast2019

# Main-A: monot5 rerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.train.cqe.trec \
    --scores monot5-probs/rerank_cast2019/ablation_train/cast2019.train.cqe.rerank.txt.probs \
    --reranked runs/cast2019/cast2019.train.cqe.rerank.trec \
    --topk 1000 \
    --prefix monot5 
# Main-A: Convrerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.train.cqe.trec \
    --scores monot5-probs/rerank_cast2019/ablation_train/cast2019.train.cqe.conv.rerank.txt.probs \
    --reranked runs/cast2019/cast2019.train.cqe.conv.rerank.trec \
    --topk 1000 \
    --prefix conv-monot5 


echo 'Run, nDCG@3, nDCG@5, nDCG@10, nDCG@1000'
for run in runs/cast2019/*train*trec;do
    echo -n ${run##*cast2019.}','
    ./trec_eval-9.0.7/trec_eval \
        -m ndcg_cut.3,5,10,1000 \
        data/cast2019/train_topics_mod.qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done

echo 'Run, nDCG@3, nDCG@5, nDCG@10, nDCG@1000'
for run in runs/cast2019/*all*trec;do
    echo -n ${run##*cast2019.}','
    ./trec_eval-9.0.7/trec_eval \
        -m ndcg_cut.3,5,10,1000 \
        data/cast2019/2019qrels_all.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done
