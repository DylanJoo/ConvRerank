import argparse
import collections
import json
import os
import re
from utils import load_collections

def convert_trecweb_to_jsonl(path_eval, path_manual, path_output, path_corpus):
    # load manual rewritten query manually
    data = json.load(open(path_eval, 'r'))
    manual = dict(tuple(data.strip().split('\t')) \
            for data in open(path_manual, 'r').readlines())
    collections = load_collections(dir=path_corpus)

    with open(path_output, 'w') as f:
        for topic_i, topic in enumerate(data):
            topic_id = topic['number']

            # place holder for history
            history = {"utterances": []}
            for turn_i, turn in enumerate(topic['turn']):
                turn_id = turn['number']

                if (turn_i+1) != turn['number']:
                    print(f"Query id correction: {topic_id}-{turn_id} to {topic_id}-{turn_idx+1}")
                    turn_id = turn_idx + 1

                # information
                utterance = turn['raw_utterance'].strip()
                manual_rewritten = manual[f"{topic_id}_{turn_id}"]
                
                # output
                f.write(json.dumps({
                    'id': f"{topic_id}_{turn_id}",
                    'utterance': utterance,
                    'manual': manual_rewritten.
                    'history_utterances': history['utterances']
                }) +'\n')

                # history
                history['utterances'].append(utterance)

if __name__ == '__main__':

    convert_trecweb_to_jsonl(
            path_eval='data/cast2019/evaluation_topics_v1.0.json',
            path_manual='data/cast2019/evaluation_topics_annotated_resolved_v1.0.tsv',
            path_output='data/cast2019/cast2019.eval.jsonl',
            path_corpus='/tmp2/jhju/datasets/cast2019/'
    )

