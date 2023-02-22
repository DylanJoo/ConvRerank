mkdir -p runs/cast2020
candidate=$1

# Main-A: monot5 rerank & Convrerank
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2020/cast2020.eval.cqe.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.rerank.trec \
    --topk 100 \
    --prefix monot5 
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.cqe.trec \
    --scores monot5-probs/rerank_cast2020/cast2020.eval.cqe.conv.rerank.$candidate.txt.probs \
    --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.trec \
    --topk 100 \
    --prefix conv-monot5 

# Main-B: View
for view in singleview0 singleview1 reverseview;do
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.cqe.trec \
        --scores monot5-probs/rerank_cast2020/ablation_views/cast2020.eval.cqe.conv.rerank.$view.txt.probs \
        --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.$view.trec \
        --topk 100 \
        --prefix conv-monot5 
done

# Main-C: TopK
for topk in t10 t20 t30 t40 t50 t100;do
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.cqe.trec \
        --scores monot5-probs/rerank_cast2020/cast2020.eval.cqe.conv.rerank.$topk.txt.probs \
        --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.$topk.trec \
        --topk 100 \
        --prefix conv-monot5 
done

# Ablation-B: monot5 rerank zero-shot (It can also be the main)
# for first_stage in cqe t5-cqe t5-dpr;do
for first_stage in cqe;do
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.$first_stage.trec \
        --scores monot5-probs/rerank_cast2020/ablation_zeroshot/cast2020.eval.$first_stage.zs.rerank.txt.probs \
        --reranked runs/cast2020/cast2020.eval.$first_stage.zs.rerank.trec \
        --topk 100 \
        --prefix zs-monot5 
done

# Ablation-A: different first-stage candidates
for first_stage in cqe-hybrid manual.dpr;do
    # Convrerank
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.$first_stage.trec \
        --scores monot5-probs/rerank_cast2020/ablation_firststage/cast2020.eval.$first_stage.conv.rerank.txt.probs \
        --reranked runs/cast2020/cast2020.eval.$first_stage.conv.rerank.trec \
        --topk 100 \
        --prefix conv-monot5 
    # monot5 
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.$first_stage.trec \
        --scores monot5-probs/rerank_cast2020/ablation_firststage/cast2020.eval.$first_stage.rerank.txt.probs \
        --reranked runs/cast2020/cast2020.eval.$first_stage.rerank.trec \
        --topk 100 \
        --prefix monot5 
done
# Convrerank
first_stage=hqe
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.$first_stage.trec \
    --scores monot5-probs/rerank_cast2020/ablation_firststage/cast2020.eval.$first_stage.conv.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.$first_stage.conv.rerank.trec \
    --topk 1000 \
    --prefix conv-monot5 
# monot5 
python3 tools/rerank_runs.py \
    --baseline runs/cast2020/cast2020.eval.$first_stage.trec \
    --scores monot5-probs/rerank_cast2020/ablation_firststage/cast2020.eval.$first_stage.rerank.txt.probs \
    --reranked runs/cast2020/cast2020.eval.$first_stage.rerank.trec \
    --topk 1000 \
    --prefix monot5 

# Ablation-C: scaling model size
for model_size in large 3B;do
    # Convrerank
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.cqe.trec \
        --scores monot5-probs/rerank_cast2020/ablation_size/cast2020.eval.cqe.conv.rerank.$model_size.txt.probs \
        --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.$model_size.trec \
        --topk 100 \
        --prefix conv-monot5 
    # monot5
    python3 tools/rerank_runs.py \
        --baseline runs/cast2020/cast2020.eval.cqe.trec \
        --scores monot5-probs/rerank_cast2020/ablation_size/cast2020.eval.cqe.rerank.$model_size.txt.probs \
        --reranked runs/cast2020/cast2020.eval.cqe.rerank.$model_size.trec \
        --topk 100 \
        --prefix monot5 
done

echo 'Run, nDCG@3,nDCG@100'
for run in runs/cast2020/*eval*.trec;do
    echo -n ${run##*cast2020.}','
    ./trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.3,100 \
        data/cast2020/2020qrels.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done
echo 'Run, Recall@100'
for first_stage in hqe cqe cqe-hybrid manual.dpr;do
    run=runs/cast2020/cast2020.eval.$first_stage.trec
    echo -n $first_stage','
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        data/cast2020/2020qrels.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done
