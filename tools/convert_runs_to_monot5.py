import argparse
from tqdm import tqdm
import collections
from datasets import Dataset
from utils import load_topics, load_collections, load_runs, normalized

def convert_runs_to_monot5(args):

    # load triplet
    queries = load_topics(args.topics)
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
            if args.preserved_turns>0 and len(query['history_utterances']) > args.window_size:
                context = query['history_utterances'][:args.preserved_turns] + \
                        query['history_utterances'][-(args.window_size-args.preserved_turns):]
            else:
                context = query['history_utterances'][-args.window_size:]
            if args.add_topic and query['topic'] != '':
                context = [query['topic']] + context
            data['context'].append(context)
            try:
                data['passage'].append(normalized(passages[pid]))
            except:
                data['passage'].append("")

    dataset = Dataset.from_dict(data)

    # output examples (to be inferenced)
    with open(args.output, 'w') as f:
        for data in tqdm(dataset):
            if args.conversational: 
                if len(data['context'])>0:
                    sep_token = " <extra_id_10> "
                    context = sep_token.join(data['context']).strip()
                    example = f"Query: {data['utterance']} Context: {context} Document: {data['passage']} Relevant:"
                else:
                    example = f"Query: {data['utterance']} Document: {data['passage']} Relevant:"
            else:
                query_input = data[f'args.query_input']
                example = f"Query: {query_input} Document: {data['passage']} Relevant:"
            f.write(example+'\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--topics", type=str, required=True,)
    parser.add_argument("--collection", type=str, required=True,)
    parser.add_argument("--query_input", type=str, default='rewrite')
    # Reranking conditions
    parser.add_argument("--output", type=str, default="")
    # Conversataion conditions
    parser.add_argument("--conversational", default=False, action='store_true')
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--preserved_turns", type=int, default=0)
    parser.add_argument("--add_topic", default=False, action='store_true')
    args = parser.parse_args()

    convert_runs_to_monot5(args)
