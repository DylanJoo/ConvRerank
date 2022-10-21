import os
import argparse
import random
import collections
import json
from utils import load_runs, load_collections, load_topics, normalized

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="data/canard/train.queries.jsonl", type=str)
parser.add_argument("--run_teacher", \
        default="spr/runs/cast20.canard.train.answer+rewrite.spr.top1000.trec", type=str)
parser.add_argument("--run_student", \
        default="spr/runs/cast20.canard.train.rewrite.spr.top1000.trec", type=str)
parser.add_argument("--output", default="convir/convir.train.convrerank.jsonl", type=str)
parser.add_argument("--topk_pool", type=int, default=200)
parser.add_argument("--topk_positive", type=int, default=None)
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--window_size", type=int, default=0)
parser.add_argument("-collections", "--collections", type=str, default="data/trec-car+marco-psg/")
# parser.add_argument("--multiview", action='store_true', default=False)
args = parser.parse_args()


queries = load_topics(args.topic)
run_student = load_runs(args.run_student)
run_teacher = load_runs(args.run_teacher)
passages = load_collections(dir=args.collections)
assert len(run_student) == len(run_teacher), "Inconsistent number of queries"

# set seed
random.seed(123)
# count = collections.defaultdict(list)
fout = open(args.output, 'w')

with open(args.topic) as topic:
    for i, line in enumerate(topic):
        query_dict = json.loads(line.strip())
        qid = query_dict['id']
        ranklist_teacher = [docid for docid in run_teacher[qid][:args.topk_pool]]
        ranklist_student = [docid for docid in run_student[qid][:args.topk_pool]]

        # de-noising boundary of positive/negative
        in_pool = [docid for docid in ranklist_student if docid in ranklist_teacher]
        out_pool = [docid for docid in ranklist_student if docid not in ranklist_teacher]
        positive_pool = (in_pool+out_pool)[:args.topk_positive]
        negative_pool = (in_pool+out_pool)[args.topk_positive:]

        # sampling positives and negatives
        psg_ids_pos = doc_pool_random_sampling(positive_pool, args.n)
        psg_ids_neg = doc_pool_random_sampling(negative_pool, args.n)
        
        q = normalized(query_dict['utterance'])
        c_t = "|".join(query_dict['history_topic'])
        c_u = query_dict['history_utterances'][-args.window_size:]
        c = normalized("|".join([c_t] + c_u))

        for j, (psg_id_pos, psg_id_neg) in enumerate(zip(psg_ids_pos, psg_ids_neg)):
            d_pos = normalized(passages[psg_id_pos])
            d_neg = normalized(passages[psg_id_neg])
            fout.write(f"Query: {q} Context: {c} Document: {d_pos} Relevant:\ttrue\n")
            fout.write(f"Query: {q} Context: {c} Document: {d_neg} Relevant:\tfalse\n")

    if i % 10000 == 0:
        print(f"{i} convir queries finished...")

fout.close()

print("DONE")

