export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# run sparse search for reformuated query (view0)
python3 tools/sparse_retrieval.py \
    --k 1000 --k1 0.82 --b 0.68 \
    --index /tmp2/jhju/indexes/cast2020_psg \
    --topics data/canard/train.jsonl \
    --query rewrite \
    --output runs/cast20.canard.train.view0.bm25.top1000.trec &

# run sparse search for reformuated query plus answer (view1)
python3 tools/sparse_retrieval.py \
    --k 1000 --k1 0.82 --b 0.68 \
    --index /tmp2/jhju/indexes/cast2020_psg \
    --topics data/canard/train.jsonl \
    --query rewrite+answer \
    --output runs/cast20.canard.train.view1.bm25.top1000.trec

# run sparse search for utterance (viewx)
python3 tools/sparse_retrieval.py \
    --k 1000 --k1 0.82 --b 0.68 \
    --index /tmp2/jhju/indexes/cast2020_psg \
    --topics data/canard/train.jsonl \
    --query utterance \
    --output runs/cast20.canard.train.viewx.bm25.top1000.trec
