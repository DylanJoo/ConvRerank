# evaluation on cast 2020
echo 'Run, R@100, R@1000, nDCG@3, nDCG@5, nDCG@500,nDCG@1000'
for run in runs/cast2020/*;do
    echo -n ${run##*cast2020.}','
    ./trec_eval-9.0.7/trec_eval \
        -m recall.100,1000 -m ndcg_cut.3,5,500,1000 \
        data/cast2020/2020qrels.txt $run | cut -f3 | sed ':a; N; $!ba; s/\n/,/g'
done
