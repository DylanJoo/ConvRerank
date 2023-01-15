# evaluation on cast 2020
for runs in runs/cast2020/*;do
    echo '| Run name | R@100  | nDCG@3/5/500/1000 |'
    echo '|----------|--------|--------_---------|'

    echo -n ${runs} '| mDPR |';
    run=runs/run.miracl.mdpr.lang.dev/run.miracl.mdpr.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
done
