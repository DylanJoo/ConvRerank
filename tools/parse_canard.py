import json
import argparse
import collections
import os

def main(args):

    canard = json.load(open(args.path_canard, 'r'))

    with open(args.path_output, 'w') as fout:

        for i, dict_canard in enumerate(canard):

            query_dict = { }
            quac_id = f"{dict_canard['QuAC_dialog_id']}_q#{dict_canard['Question_no']-1}"
            assert QUAC_ANS[quac_id]['Question'] == dict_canard['Question'], 'Mismatched'

            query_dict['id'] = quac_id
            query_dict['answer'] = QUAC_ANS[quac_id]['Answer'].strip()
            query_dict['utterance'] = QUAC_ANS[quac_id]['Question'].strip()
            query_dict['rewrite'] = dict_canard['Rewrite'].strip()
            query_dict['history_topic'] = dict_canard['History'][:2]
            query_dict['history_utterances'] = \
                    [c.strip() for i, c in enumerate(dict_canard['History'][2:]) if i % 2 == 0]
            query_dict['history_responses'] = \
                    [c.strip() for i, c in enumerate(dict_canard['History'][2:]) if i % 2 == 1]

            fout.write(json.dumps(query_dict) + '\n')

            if i % 10000 == 0:
                print("{} finished...".format(i))

def parse_quac(dir):
    quac = json.load(open(os.path.join(dir, 'train_v0.2.json'), 'r'))['data'] + \
            json.load(open(os.path.join(dir, 'val_v0.2.json'), 'r'))['data'] 
    data = collections.defaultdict(dict)
    for topic in quac:
        i = 0
        for turn in topic['paragraphs'][0]['qas']:
            turn_id = turn['id']
            if turn_id.split("q#")[1] != str(i):
                print("[FIX] Incorrent turn numbers found, fix QuAC turn number")
                turn_id = f'{turn_id.split("q#")[0]}q#{i}'

            if turn['id'] == "C_2ca59977d66d4742939232f443ceda41_1_q#6":
                print(f"[FIX] Ambiguous question: {turn['question']}, ignore this turn.")
                i -= 1
            else:
                data[turn_id] = {
                        'Answer': turn['orig_answer']['text'].replace("CANNOTANSWER", ""), 
                        'Question': turn['question']
                }
            i += 1
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-canard", "--path_canard", default="data/canard/train.json", type=str)
    parser.add_argument("-output", "--path_output", default='data/canard/train.jsonl', type=str)
    parser.add_argument("-quac", "--dir_quac", default="data/quac/", type=str)
    args = parser.parse_args()

    QUAC_ANS = parse_quac(args.dir_quac)

    main(args)
