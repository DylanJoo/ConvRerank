mkdir -p monot5/cast2020/
# convert to monot5 (ablation)
# for first_stage in cqe t5-cqe t5-dpr;do
#     run=runs/cast2020/cast2020.eval.${first_stage}.trec
#     file=${run##*/}
#     python3 tools/convert_runs_to_monot5.py \
#       --run $run \
#       --topic data/cast2020/cast2020.eval.jsonl \
#       --collection /tmp2/jhju/datasets/cast2020 \
#       --output monot5/${file/trec/rerank\.txt}
# done &

# convert to conversational monot5
for first_stage in cqe t5-cqe t5-dpr;do
    run=runs/cast2020/cast2020.eval.${first_stage}.trec
    file=${run##*/}
    python3 tools/convert_runs_to_monot5.py \
      --run $run \
      --topic data/cast2020/cast2020.eval.jsonl \
      --collection /tmp2/jhju/datasets/cast2020 \
      --output monot5/cast2020/${file/trec/conv.rerank\.txt} \
      --conversational
done
