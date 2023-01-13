import os
import argparse
import random
import collections
import json
from tqdm import tqdm
from utils import load_runs, load_collections, load_topics, normalized, doc_pool_random_sampling


def conversational_pseudo_labeling(args):
    # load topics, two runs and passage collections
    queries = load_topics(args.topic)
    run0 = load_runs(args.run0)
    run1 = load_runs(args.run1)
    passages = load_collections(dir=args.collections)
    assert len(run0) == len(run1), f"Inconsistent number of queries: len(run0) and len(run1)"
    print('resource loaded')

    random.seed(123)
    with open(args.output, 'w') as fout, \
         open(args.output.replace('txt', 'stats'), 'w') as fstat:

        for qid in tqdm(queries):
            turn_i = qid.split('#')[-1]
            query = queries[qid]
            # view0 and view1 ranked list
            ranklist_view0 = [docid for docid in run0[qid][:args.topk_pool]]
            ranklist_view1 = [docid for docid in run1[qid][:args.topk_pool]]

            # Pseudo positive/negative based on view0
            ## in_pool: collect the easy positive candidates
            ## out_pool: collect the harder negative candidates
            # [CONCERN] out_pool and in_pool order
            in_pool = [docid for docid in ranklist_view0 if docid in ranklist_view1]
            out_pool = [docid for docid in ranklist_view0 if docid not in ranklist_view1]
            positive_pool = (in_pool+out_pool)[:args.topk_positive]
            negative_pool = (in_pool+out_pool)[args.topk_positive:]

            # sampling positives and negatives, n is the number of triplets for each topic
            psg_ids_pos = doc_pool_random_sampling(positive_pool, args.n)
            psg_ids_neg = doc_pool_random_sampling(negative_pool, args.n)
            
            # the conversational query
            ## 0: current turn of information need
            ## c_t: detail topic of historical information need
            ## c_u: historical information need
            ## w: the window size of historical utterance
            a = normalized(query['answer'])
            q = normalized(query['utterance'])
            c_t = "|".join(query['history_topic'])
            c_u = query['history_utterances'][-args.window_size:]
            c = normalized("|".join([c_t] + c_u))

            for psg_id_pos, psg_id_neg in zip(psg_ids_pos, psg_ids_neg):
                d_pos = normalized(passages[psg_id_pos])
                d_neg = normalized(passages[psg_id_neg])
                fout.write(f"Query: {q} Context: {c} Document: {d_pos} Relevant:\ttrue\n")
                fout.write(f"Query: {q} Context: {c} Document: {d_neg} Relevant:\tfalse\n")

            # append stat
            fstat.write(f"{turn_i}, {len(in_pool)}, {len(out_pool)}, {q}, {a}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="data/canard/train.jsonl", type=str)
    parser.add_argument("--run0", type=str, default="cast20.canard.train.view0.bm25.top1000.trec")
    parser.add_argument("--run1", type=str, default="cast20.canard.train.view1.bm25.top1000.trec")
    parser.add_argument("--output", default="data/canard4ir/convir.train.view1-0.jsonl", type=str)
    parser.add_argument("--topk_pool", type=int, default=200)
    parser.add_argument("--topk_positive", type=int, default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--collections", type=str, default="/tmp2/jhju/datasets/cast2020")
    args = parser.parse_args()

    conversational_pseudo_labeling(args)

