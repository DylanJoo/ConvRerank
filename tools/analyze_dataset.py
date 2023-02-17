import os
import argparse
import random
import collections
import json
from tqdm import tqdm
from utils import load_runs, load_collections, load_topics, normalized, doc_pool_random_sampling

def dualview_pseudo_labeling(args):
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
            topic_i = qid.split('#')[0]
            turn_i = qid.split('#')[-1]
            query = queries[qid]
            # view0 and view1 ranked list
            ranklist_view0 = [docid for docid in run0[qid][:args.topk_pool]]
            ranklist_view1 = [docid for docid in run1[qid][:args.topk_pool]]

            # Pseudo positive/negative based on view0
            ## in_pool: collect the easy positive candidates
            ## out_pool: collect the harder negative candidates
            # [CONCERN] out_pool and in_pool order
            in_pool = [docid for i, docid in enumerate(ranklist_view0) if docid in ranklist_view1]
            out_pool = [docid for i, docid in enumerate(ranklist_view0) if docid not in ranklist_view1]
            positive_pool = (in_pool+out_pool)[:args.topk_positive]
            negative_pool = (in_pool+out_pool)[args.topk_positive:]

            # sampling positives and negatives, n is the number of triplets for each topic
            psg_ids_pos = doc_pool_random_sampling(positive_pool, args.n)
            psg_ids_neg = doc_pool_random_sampling(negative_pool, args.n)

            # analyzing
            distilled_positives = [(i, docid) for i, docid in enumerate(ranklist_view0) \
                    if docid in in_pool]
            boundary = distilled_positives[-1][0] if len(distilled_positives) > 0 else -1
            deposited_negatives = [(i, docid) for i, docid in enumerate(ranklist_view0) \
                    if (docid in out_pool and i < boundary)]

            # the conversational query
            ## 0: current turn of information need
            ## c_t: detail topic of historical information need
            ## c_u: historical information need
            ## w: the window size of historical utterance
            sep_token = " | "
            c_t = sep_token.join(query['history_topic'])
            c_u = query['history_utterances'][-args.window_size:]
            c = normalized(sep_token.join([c_t] + c_u))

            a = normalized(query['answer'])
            q_star = normalized(query['rewrite'])

            fout.write("#"*10+'\n')
            fout.write(f"Query: {q_star}\nHistory: {c}\nAnswer: {a}\n")
            for (i, docid) in deposited_negatives[:2]:
                doc = passages[docid]
                fout.write(f"FN {i}, {doc}\n")
            for (i, docid) in distilled_positives[:2]:
                doc = passages[docid]
                fout.write(f"TP {i}, {doc}\n")
            fout.write("\n")

            # append stat
            fstat.write(f"{topic_i}, {turn_i}, {len(in_pool)}, {len(out_pool)}, {len(distilled_positives)}, {len(deposited_negatives)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="data/canard/train.jsonl", type=str)
    parser.add_argument("--run0", type=str, default="cast20.canard.train.view0.bm25.top1000.trec")
    parser.add_argument("--run1", type=str, default=None)
    parser.add_argument("--output", default="data/canard4ir/analysis.txt", type=str)
    parser.add_argument("--topk_pool", type=int, default=200)
    parser.add_argument("--topk_positive", type=int, default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--collections", type=str, default="/tmp2/jhju/datasets/cast2020")
    parser.add_argument("--singleview", action="store_true", default=False)
    args = parser.parse_args()

    dualview_pseudo_labeling(args)

