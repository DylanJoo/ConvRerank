mkdir -p runs/cast2019

# Main-A: monot5 rerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.rerank.txt.probs \
    --reranked runs/cast2019/cast2019.eval.cqe.rerank.trec \
    --topk 1000 \
    --prefix monot5 
# Main-A: Convrerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.conv.rerank.t20.txt.probs \
    --reranked runs/cast2019/cast2019.eval.cqe.conv.rerank.t20.trec \
    --topk 1000 \
    --prefix conv-monot5 
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.conv.rerank.extra.txt.probs \
    --reranked runs/cast2019/cast2019.eval.cqe.conv.rerank.extra.trec \
    --topk 1000 \
    --prefix conv-monot5 

# # Main-B: View
for view in singleview0 singleview1 reverseview;do
    python3 tools/rerank_runs.py \
        --baseline runs/cast2019/cast2019.eval.cqe.trec \
        --scores monot5-probs/rerank_cast2019/ablation_$view/cast2019.eval.cqe.conv.rerank.txt.probs \
        --reranked runs/cast2019/cast2019.eval.cqe.conv.rerank.$view.trec \
        --topk 1000 \
        --prefix conv-monot5 
done

# Main-C: TopK
for topk in t3 t10 t30 j20;do
    python3 tools/rerank_runs.py \
        --baseline runs/cast2019/cast2019.eval.cqe.trec \
        --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.conv.rerank.$topk.txt.probs \
        --reranked runs/cast2019/cast2019.eval.cqe.conv.rerank.$topk.trec \
        --topk 1000 \
        --prefix conv-monot5 
done

# # Ablation-A: different first-stage candidates
for first_stage in t5-cqe t5-dpr;do
    # Convrerank
    python3 tools/rerank_runs.py \
        --baseline runs/cast2019/cast2019.eval.$first_stage.trec \
        --scores monot5-probs/rerank_cast2019/ablation_firststage/cast2019.eval.$first_stage.conv.rerank.txt.probs \
        --reranked runs/cast2019/cast2019.eval.$first_stage.conv.rerank.trec \
        --topk 1000 \
        --prefix conv-monot5 
    # monot5 
    python3 tools/rerank_runs.py \
        --baseline runs/cast2019/cast2019.eval.$first_stage.trec \
        --scores monot5-probs/rerank_cast2019/ablation_firststage/cast2019.eval.$first_stage.rerank.txt.probs \
        --reranked runs/cast2019/cast2019.eval.$first_stage.rerank.trec \
        --topk 1000 \
        --prefix conv-monot5 
done

# Ablation-B: monot5 rerank zero-shot (It can also be the main)
python3 tools/rerank_runs.py \
    --baseline runs/cast2019/cast2019.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2019/cast2019.eval.cqe.zs.rerank.txt.probs \
    --reranked runs/cast2019/cast2019.eval.cqe.zs.rerank.trec \
    --topk 1000 \
    --prefix zs-monot5 
# for first_stage in t5-cqe t5-dpr;do
#     # monot5 (zero-shot)
#     python3 tools/rerank_runs.py \
#         --baseline runs/cast2019/cast2019.eval.$first_stage.trec \
#         --scores monot5-probs/rerank_cast2019/ablation_firststage/cast2019.eval.$first_stage.zs.rerank.txt.probs \
#         --reranked runs/cast2019/cast2019.eval.$first_stage.zs.rerank.trec \
#         --topk 1000 \
#         --prefix zs-monot5 
# done

# Ablation-C: scaling model size
# for model_size in large 3B;do
#     # Convrerank
#     python3 tools/rerank_runs.py \
#         --baseline runs/cast2019/cast2019.eval.cqe.trec \
#         --scores monot5-probs/rerank_cast2019/ablation_$model_size/cast2019.eval.cqe.conv.rerank.txt.probs \
#         --reranked runs/cast2019/cast2019.eval.cqe.conv.rerank.$model_size.trec \
#         --topk 1000 \
#         --prefix conv-monot5 
#     # monot5
#     python3 tools/rerank_runs.py \
#         --baseline runs/cast2019/cast2019.eval.cqe.trec \
#         --scores monot5-probs/rerank_cast2019/ablation_$model_size/cast2019.eval.cqe.rerank.txt.probs \
#         --reranked runs/cast2019/cast2019.eval.cqe.rerank.$model_size.trec \
#         --topk 1000 \
#         --prefix monot5 
# done

# Ablation-D: different warm up stage
# for warmup in from10k from0k;do
#     python3 tools/rerank_runs.py \
#         --baseline runs/cast2019/cast2019.eval.cqe.trec \
#         --scores monot5-probs/rerank_cast2019/ablation_warmup/cast2019.eval.cqe.conv.rerank.$warmup.txt.probs \
#         --reranked runs/cast2019/cast2019.eval.cqe.conv.rerank.$warmup.trec \
#         --topk 1000 \
#         --prefix conv-monot5 
# done


echo 'Run, nDCG@3, nDCG@5, nDCG@10, nDCG@1000'
for run in runs/cast2019/*.trec;do
    echo -n ${run##*cast2019.}','
    ./trec_eval-9.0.7/trec_eval \
        -m ndcg_cut.3,5,10,1000 \
        data/cast2019/2019qrels.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done
