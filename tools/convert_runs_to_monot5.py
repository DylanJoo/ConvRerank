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
        query_text = queries[qid]
        for pid in runs[qid]:
            data['qid'].append(qid)
            data['pid'].append(pid)
            data['query'].append(query_text['rewrite'])
            data['passage'].append(normalized(passages[pid]))

    dataset = Dataset.from_dict(data)

    # data loader
    # datacollator = DataCollatorFormonoT5(
    #         return_text=True,
    # )
    # dataloader = DataLoader(
    #         dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False, # cannot be shuffle
    #         collate_fn=datacollator
    # )

    # output examples (to be inferenced)
    with open(args.output, 'w') as f:
        for data in tqdm(dataset):
            example = f"Query: {data['query']} Document: {data['passage']} Relevant:"
            f.write(example+'\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--topic", type=str, required=True,)
    parser.add_argument("--collection", type=str, required=True,)
    # Reranking conditions
    parser.add_argument("--output", type=str, default="")
    # parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    convert_runs_to_monot5(args)
