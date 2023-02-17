mkdir -p monot5/cast2019

# convert to monot5 
for first_stage in hqe;do
    run=runs/cast2019/cast2019.eval.${first_stage}.trec
    file=${run##*/}
    python3 tools/convert_runs_to_monot5.py \
      --run $run \
      --topics data/cast2019/cast2019.eval.jsonl \
      --collection /tmp2/jhju/datasets/cast2020 \
      --output monot5/cast2019/${file/trec/rerank.txt}
done &

# convert to conversational monot5
for first_stage in hqe;do
    run=runs/cast2019/cast2019.eval.${first_stage}.trec
    file=${run##*/}
    python3 tools/convert_runs_to_monot5.py \
      --run $run \
      --topics data/cast2019/cast2019.eval.jsonl \
      --collection /tmp2/jhju/datasets/cast2020 \
      --output monot5/cast2019/${file/trec/conv.rerank.txt} \
      --conversational
done

# using train set
# run=runs/cast2019/cast2019.train.cqe.trec
# file=${run##*/}
# python3 tools/convert_runs_to_monot5.py \
#   --run $run \
#   --topics data/cast2019/cast2019.train.jsonl \
#   --collection /tmp2/jhju/datasets/cast2020 \
#   --output monot5/cast2019/${file/trec/rerank\.txt}
#
# python3 tools/convert_runs_to_monot5.py \
#   --run $run \
#   --topics data/cast2019/cast2019.train.jsonl \
#   --collection /tmp2/jhju/datasets/cast2020 \
#   --output monot5/cast2019/${file/trec/conv.rerank\.txt} \
#   --conversational

