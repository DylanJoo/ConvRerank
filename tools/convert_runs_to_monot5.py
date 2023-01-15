"""
Convert runs(.trec) into monot5 input
with the format: Query: <q> Document: <d> Relevant:
"""
import argparse
from tqdm import tqdm
import collections
from torch.utils.data import DataLoader
from datasets import Dataset
import socket

if 'cfda' in socket.gethostname():
    from utils import load_runs, load_collections, load_topics, normalized
else:
    from utils_gcp import load_runs, load_collections, load_topics, normalized


def convert_runs_to_monot5(args):

    # load triplet
    queries = load_topics(args.topic)
    passages = load_collections(dir=args.collection)
    runs = load_runs(args.run)

    # prepare dataset
    data = collections.defaultdict(list)
    for qid in runs:
        query = queries[qid]
        for pid in runs[qid]:
            data['qid'].append(qid)
            data['pid'].append(pid)
            data['rewrite'].append(query['rewrite'])
            data['manual'].append(query['manual'])
            data['utterance'].append(query['utterance'])
            data['context'].append(query['history_utterances'][-args.window_size:])
            data['passage'].append(normalized(passages[pid]))

    dataset = Dataset.from_dict(data)

    # output examples (to be inferenced)
    with open(args.output, 'w') as f:
        for data in tqdm(dataset):
            if args.conversational:
                example = f"Query: {data['utterance']} Context: {data['context']} Document: {data['passage']} Relevant:"
            else:
                example = f"Query: {data['rewrite']} Document: {data['passage']} Relevant:"
            f.write(example+'\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--topic", type=str, required=True,)
    parser.add_argument("--collection", type=str, required=True,)
    # Reranking conditions
    parser.add_argument("--output", type=str, default="")
    # Conversataion conditions
    parser.add_argument("--conversational", default=False, action='store_true')
    parser.add_argument("--window_size", type=int, default=3)
    args = parser.parse_args()

    convert_runs_to_monot5(args)
