# cast 2020
# parse the cast2020 topic
# python3 tools/parse_cast2020.py
# convert to monot5
for run in runs/cast2020/*;do
    file=${run##*/}
    python3 tools/convert_runs_to_monot5.py \
      --run $run \
      --topic data/cast2020/cast2020.eval.jsonl \
      --collection /tmp2/jhju/datasets/cast2020 \
      --output monot5/${file/trec/rerank\.trec} \
      --conversational
done
