year=$1
evalset=$2
if [[ "$year" = "2019" ]]
then
    truth=2019qrels.txt
    if [[ "$evalset" = "train" ]]
    then
        truth=train_topics_mod.qrel
    fi
fi
if [[ "$year" = "2020" ]]
then
    truth=2020qrels.txt
fi

# baseline
trec_eval-9.0.7/trec_eval -q \
    -m ndcg_cut.3 \
    data/cast${year}/$truth \
    runs/cast${year}/cast${year}.${evalset}.cqe.trec | cut -f2,3 > $year.result0.txt

# monot5
trec_eval-9.0.7/trec_eval -q \
    -m ndcg_cut.3 \
    data/cast${year}/$truth \
    runs/cast${year}/cast${year}.${evalset}.cqe.rerank.trec | cut -f2,3 > $year.resultx.txt

# convrerank
trec_eval-9.0.7/trec_eval -q \
    -m ndcg_cut.3 \
    data/cast${year}/$truth \
    runs/cast${year}/cast${year}.${evalset}.cqe.conv.rerank.t40.trec | cut -f2,3 > $year.resulty.txt

python3 tools/analyze_results.py \
    --result0 $year.result0.txt \
    --resultx $year.resultx.txt \
    --resulty $year.resulty.txt \
    --runx runs/cast${year}/cast${year}.${evalset}.cqe.rerank.trec \
    --runy runs/cast${year}/cast${year}.${evalset}.cqe.conv.rerank.t40.trec

rm $year.result0.txt
rm $year.resultx.txt
rm $year.resulty.txt
