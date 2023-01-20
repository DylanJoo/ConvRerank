candidates=$1
# rerank
# python3 tools/rerank_runs.py \
#     --baseline runs/cast2020/cast2020.eval.$candidates.trec \
#     --scores monot5-probs/cast2020.eval.$candidates.rerank.txt.probs \
#     --reranked runs/cast2020/cast2020.eval.$candidates.rerank.trec \
#     --topk 1000 \
#     --prefix monot5 
#
# # conv rerank
# python3 tools/rerank_runs.py \
#     --baseline runs/cast2020/cast2020.eval.$candidates.trec \
#     --scores monot5-probs/cast2020.eval.$candidates.conv.rerank.txt.probs \
#     --reranked runs/cast2020/cast2020.eval.$candidates.conv.rerank.trec \
#     --topk 1000 \
#     --prefix conv-monot5 
#
# ## ablations study 
# for ablation in from10k singleview0 singleview1 reverseview dev;do
#     python3 tools/rerank_runs.py \
#         --baseline runs/cast2020/cast2020.eval.cqe.trec \
#         --scores monot5-probs/cast2020.eval.cqe.conv.rerank.ablation_${ablation}.probs \
#         --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.ablation_${ablation}.trec \
#         --topk 1000 \
#         --prefix conv-monot5-ablation
# done

## ablation study (zeroshot)
# for ablation in t5-cqe cqe;do
#     python3 tools/rerank_runs.py \
#         --baseline runs/cast2020/cast2020.eval.${ablation}.trec \
#         --scores monot5-probs/cast2020.eval.${ablation}.rerank.ablation_zeroshot.probs \
#         --reranked runs/cast2020/cast2020.eval.${ablation}.rerank.ablation_zeroshot.trec \
#         --topk 1000 \
#         --prefix monot5-ablation
# done

## ablation study (large)
# python3 tools/rerank_runs.py \
#     --baseline runs/cast2020/cast2020.eval.cqe.trec \
#     --scores monot5-probs/cast2020.eval.cqe.conv.rerank.ablation_large.probs \
#     --reranked runs/cast2020/cast2020.eval.cqe.conv.rerank.ablation_large.trec \
#     --topk 1000 \
#     --prefix conv-monot5-large
#
# python3 tools/rerank_runs.py \
#     --baseline runs/cast2020/cast2020.eval.cqe.trec \
#     --scores monot5-probs/cast2020.eval.cqe.rerank.ablation_large.probs \
#     --reranked runs/cast2020/cast2020.eval.cqe.rerank.ablation_large.trec \
#     --topk 1000 \
#     --prefix monot5-large
