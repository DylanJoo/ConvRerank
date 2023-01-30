import argparse
import collections
import json
import os
import re
from utils import load_collections

def convert_trecweb_to_jsonl(path_eval, path_manual, path_rewrite, path_output):
    data = json.load(open(path_eval, 'r'))
    try: 
        cqr_rewrite = dict(tuple(rewrite_data.strip().split('\t')) \
                for rewrite_data in open(path_rewrite, 'r').readlines())
    except:
        print(f"[warning] Rewritten file is not existed, \
                you can run the `run_rewrite_cast2019.sh` then, \
                and re-run this parsing.")
        cqr_rewrite = collections.defaultdict(str)

    manual = collections.defaultdict(str)
    for manual_data in open(path_manual, 'r').readlines():
        qid, qtext = manual_data.strip().split('\t')
        manual[qid] = qtext

    with open(path_output, 'w') as f:
        for topic_i, topic in enumerate(data):
            topic_id = topic['number']

            history = {"utterances": [], "responses": []}

            for turn_i, turn in enumerate(topic['turn']):
                turn_id = turn['number']

                # information
                turn_info = topic['title']
                utterance = turn['raw_utterance'].strip()
                automatic_rewritten = cqr_rewrite[f"{topic_id}_{turn_id}"]
                manual_rewritten = manual[f"{topic_id}_{turn_id}"]
                
                # output
                f.write(json.dumps({
                    'id': f"{topic_id}_{turn_id}",
                    'topic': turn_info, 
                    'utterance': utterance,
                    'rewrite': automatic_rewritten,
                    'manual': manual_rewritten,
                    'history_responses': history['responses'],
                    'history_utterances': history['utterances']
                }) +'\n')

                # history
                history['utterances'].append(utterance)
                history['responses'].append("")

if __name__ == '__main__':

    # note that the corpus "wapo" was removed in qrels
    # so, for simplicity, we use the same corpus for cast2019 and cast2020

    # Transform the cast'19 evaluation topics
    convert_trecweb_to_jsonl(
            path_eval='data/cast2019/evaluation_topics_v1.0.json',
            path_manual='data/cast2019/evaluation_topics_annotated_resolved_v1.0.tsv',
            path_rewrite='data/cast2019/cast2019.eval.rewrite.tsv',
            path_output='data/cast2019/cast2019.eval.jsonl',
    )

    # Transform the cast'20 training topics
    convert_trecweb_to_jsonl(
            path_eval='data/cast2019/train_topics_v1.0.json',
            path_manual='data/cast2019/train_topic_sample_annotated_resolved_v1.0.tsv',
            path_rewrite='data/cast2019/cast2019.train.rewrite.tsv',
            path_output='data/cast2019/cast2019.train.jsonl',
    )

