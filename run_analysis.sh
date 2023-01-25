year=$1
# baseline
trec_eval-9.0.7/trec_eval -q \
    -m ndcg_cut.3 \
    data/cast${year}/${year}qrels.txt \
    runs/cast${year}/cast${year}.eval.cqe.trec | cut -f2,3 > run0.txt

# monot5
trec_eval-9.0.7/trec_eval -q \
    -m ndcg_cut.3 \
    data/cast${year}/${year}qrels.txt \
    runs/cast${year}/cast${year}.eval.cqe.rerank.trec | cut -f2,3 > runx.txt

# convrerank
trec_eval-9.0.7/trec_eval -q \
    -m ndcg_cut.3 \
    data/cast${year}/${year}qrels.txt \
    runs/cast${year}/cast${year}.eval.cqe.conv.rerank.window8.trec | cut -f2,3 > runy.txt

python3 tools/analyze_runs.py
rm run0.txt
rm runx.txt
rm runy.txt
