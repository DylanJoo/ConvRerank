import argparse
import collections
import json
import os
import re
from utils import load_collections

def convert_trecweb_to_jsonl(path_eval, path_manual, path_automatic, path_output, path_corpus=None):
    data = json.load(open(path_eval, 'r'))
    # load manual and automatic rewritten query manually (first rewrite query using cqr)
    try: # it rewritten
        automatic = dict(tuple(automatic_data.strip().split('\t')) \
                for automatic_data in open(path_automatic, 'r').readlines())
    except:
        automatic = collections.defaultdict(str)

    manual = collections.defaultdict(str)
    for manual_data in open(path_manual, 'r').readlines():
        qid, qtext = manual_data.strip().split('\t')
        manual[qid] = qtext

    with open(path_output, 'w') as f:
        for topic_i, topic in enumerate(data):
            topic_id = topic['number']

            # place holder for history
            try: ## add global context if the data has
                history = {"utterances": [topic['title']],}
            except:
                history = {"utterances": [],}

            for turn_i, turn in enumerate(topic['turn']):
                turn_id = turn['number']

                # information
                utterance = turn['raw_utterance'].strip()
                automatic_rewritten = automatic[f"{topic_id}_{turn_id}"]
                manual_rewritten = manual[f"{topic_id}_{turn_id}"]
                
                # output
                f.write(json.dumps({
                    'id': f"{topic_id}_{turn_id}",
                    'utterance': utterance,
                    'rewrite': automatic_rewritten,
                    'manual': manual_rewritten,
                    'history_responses': "",
                    'history_utterances': history['utterances']
                }) +'\n')

                # history
                history['utterances'].append(utterance)

if __name__ == '__main__':

    # note that the corpus "wapo" was removed in qrels
    # so, for simplicity, we use the same corpus for cast2019 and cast2020

    # Transform the cast'19 evaluation topics
    convert_trecweb_to_jsonl(
            path_eval='data/cast2019/evaluation_topics_v1.0.json',
            path_manual='data/cast2019/evaluation_topics_annotated_resolved_v1.0.tsv',
            path_automatic='data/cast2019/cast2019.eval.rewrite.tsv',
            path_output='data/cast2019/cast2019.eval.jsonl',
            path_corpus='/tmp2/jhju/datasets/cast2020/'
    )

    # Transform the cast'20 training topics
    convert_trecweb_to_jsonl(
            path_eval='data/cast2019/train_topics_v1.0.json',
            path_manual='data/cast2019/train_topic_sample_annotated_resolved_v1.0.tsv',
            path_automatic='data/cast2019/cast2019.train.rewrite.tsv',
            path_output='data/cast2019/cast2019.train.jsonl',
            path_corpus='/tmp2/jhju/datasets/cast2020/'
    )

