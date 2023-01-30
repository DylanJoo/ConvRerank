import argparse
from tqdm import tqdm
import collections
from utils import load_topics

def convert_topics_to_tsv(args):

    # load triplet
    topics = load_topics(args.topics)

    # prepare dataset
    data = collections.defaultdict(list)
    with open(args.output, 'w') as f:
        for topic_id in tqdm(topics):
            topic = topics[topic_id]
            utterance = " [Q] "+ topic['utterance']
            context = topic['history_utterances']
            query = "|".join(context + [utterance])
            f.write(f"{topic_id}\t{query}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=str, required=True,)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    convert_topics_to_tsv(args)
