import argparse
import collections
import json
import os
import re
from utils import load_collections

def convert_trecweb_to_jsonl(path_eval, path_manual, path_rewrite, path_output):
    data = json.load(open(path_eval, 'r'))
    manual = json.load(open(path_manual, 'r'))
    try: # it rewritten
        cqr_rewrite = dict(tuple(rewrite_data.strip().split('\t')) \
                for rewrite_data in open(path_rewrite, 'r').readlines())
    except:
        print(f"[warning] Rewritten file is not existed, \
                \nyou can run the `run_rewrite_cast2020.sh` then, \
                \nand re-run this parsing to replace the rewritten")
        cqr_rewrite = None

    with open(path_output, 'w') as f:
        for topic_i, topic in enumerate(manual):
            topic_id = topic['number']

            # place holder for history
            history = {"utterances": [], "responses": []}
            for turn_i, turn in enumerate(topic['turn']):
                turn_id = turn['number']

                # information
                utterance = turn['raw_utterance'].strip()
                if cqr_rewrite is None:
                    automatic_rewritten = turn['automatic_rewritten_utterance'].strip()
                else:
                    automatic_rewritten = cqr_rewrite[f"{topic_id}_{turn_id}"]
                manual_rewritten = turn['manual_rewritten_utterance'].strip()
                # manual_canonical_passage_id = turn['manual_canonical_result_id']
                automatic_canonical_passage_id = data[topic_i]['turn'][turn_i]['automatic_canonical_result_id']
                
                # output
                f.write(json.dumps({
                    'id': f"{topic_id}_{turn_id}",
                    'utterance': utterance,
                    'rewrite': automatic_rewritten,
                    'manual': manual_rewritten,
                    'history_responses': history['responses'],
                    'history_utterances': history['utterances']
                }) +'\n')

                # history
                history['utterances'].append(utterance)
                history['responses'].append(automatic_canonical_passage_id)

if __name__ == '__main__':
    convert_trecweb_to_jsonl(
            path_eval='data/cast2020/2020_automatic_evaluation_topics_v1.0.json',
            path_manual='data/cast2020/2020_manual_evaluation_topics_v1.0.json',
            path_rewrite='data/cast2020/cast2020.eval.rewrite.tsv',
            path_output='data/cast2020/cast2020.eval.jsonl',
    )
