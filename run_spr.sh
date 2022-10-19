export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
mkdir -p runs

type=$1

python3 tools/sparse_retrieval.py \
  -k 1000 -k1 0.82 -b 0.68 \
  -index /tmp2/trec/cast/indexes/cast2020_psg \
  -query data/canard.train.jsonl \
  -output runs/cast20.canard.train.${type}.spr.top1000.trec \
  -qval ${type}
