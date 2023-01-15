import argparse
import collections
import json
import os
import re
from utils import load_collections

def convert_trecweb_to_jsonl(path_manual, path_automatic, path_output, path_corpus):
    manual = json.load(open(path_manual, 'r'))
    automatic = json.load(open(path_automatic, 'r'))
    collections = load_collections(dir=path_corpus)

    with open(path_output, 'w') as f:
        for topic_i, topic in enumerate(manual):
            topic_id = topic['number']

            # place holder for history
            history = {"utterances": [], "responses": []}
            for turn_i, turn in enumerate(topic['turn']):
                turn_id = turn['number']

                if (turn_i+1) != turn['number']:
                    print(f"Query id correction: {topic_id}-{turn_id} to {topic_id}-{turn_idx+1}")
                    turn_id = turn_idx + 1

                # information
                utterance = turn['raw_utterance'].strip()
                automatic_rewritten = turn['automatic_rewritten_utterance'].strip()
                manual_rewritten = turn['manual_rewritten_utterance'].strip()
                ## canonical using manual's or automatic's
                # canonical_passage_id = manual[topic_i]['turn'][turn_i]['manual_canonical_result_id']
                canonical_passage_id = automatic[topic_i]['turn'][turn_i]['automatic_canonical_result_id']
                passage_cano = collections[canonical_passage_id]
                
                # output
                f.write(json.dumps({
                    'id': f"{topic_id}_{turn_id}",
                    'utterance': utterance,
                    'rewrite': automatic_rewritten,
                    'manual': manual_rewritten,
                    'history_canonical_responses': history['responses'],
                    'history_utterances': history['utterances']
                }) +'\n')

                # history
                history['utterances'].append(utterance)
                history['responses'].append(passage_cano)

if __name__ == '__main__':
    # PATH_EVAL_COLLECTION='2020_eval_collections.json'

    convert_trecweb_to_jsonl(
            path_manual='data/cast2020/2020_manual_evaluation_topics_v1.0.json',
            path_automatic='data/cast2020/2020_automatic_evaluation_topics_v1.0.json',
            path_output='data/cast2020/cast2020.eval.jsonl',
            path_corpus='/tmp2/jhju/datasets/cast2020/' # change corpus here
    )

