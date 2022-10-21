import argparse
import collections
from reranker import monoT5
from torch.utils.data import DataLoader
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from tools.utils import load_runs, load_collections, load_topics, normalized
from datasets import Dataset
from datacollator import DataCollatorFormonoT5


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--topic", type=str, required=True,)
    parser.add_argument("--collection", type=str, required=True,)
    # Reranking conditions
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output_trec", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--prefix", type=str, default='monoT5')
    args = parser.parse_args()

    fout = open(args.output_trec, 'w')

    # load model
    model = monoT5.from_pretrained(args.model_name_or_path)
    model.set_tokenizer()
    model.set_targets()
    model.to(f'cuda:{args.gpu}')
    model.eval()

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
            data['query'].append(normalized(query_text['rewrite']))
            data['passage'].append(normalized(passages[pid]))

    dataset = Dataset.from_dict(data)

    # data loader
    datacollator = DataCollatorFormonoT5(
            tokenizer=model.tokenizer,
            padding=True,
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
            istrain=False,
    )
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False, # cannot be shuffle
            collate_fn=datacollator
    )


    # prediction
    ranking_list = collections.defaultdict(list)
    for b, batch in enumerate(dataloader):
        batch_inputs, batch_ids = batch
        output = model.predict(batch_inputs)

        true_prob = output[:, 0]
        false_prob = output[:, 1]

        for t_prob, (qid, pid) in zip(true_prob, batch_ids):
            ranking_list[qid].append((pid, t_prob))

        if b % 1000 == 0:
            print(f"{b} qp pair ranked")

    # output
    for i, (qid, candidates) in enumerate(ranking_list.items()):
        # Using true prob as score, so reverse the order.
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        for idx, (pid, t_prob) in enumerate(sorted_candidates[:args.topk]):
            example = f'{qid} Q0 {pid} {str(idx+1)} {t_prob} {args.prefix}\n'
            fout.write(example)

        if i % 100 == 0:
            print(f"{i} topics(queries) reranked")

    fout.close()
