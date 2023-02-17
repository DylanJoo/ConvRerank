from tqdm import tqdm
import argparse
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_collections

def rewrite(texts):
    """
    Function of rewrite user utterance by inputting the context (utterances and responses),
    To reformulate the current user utterance.
    """
    input_ids = tokenizer(texts, return_tensors="pt", truncation=True).input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=512, num_beams=args.beam_size)
    # rewrite_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True,)
    rewrite_text = tokenizer.decode(output_ids[0], skip_special_tokens=True,)
    return rewrite_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--model_name", type=str, default='castorini/t5-base-canard')
    parser.add_argument("--canonical", type=int, default=0)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--add_topic", action='store_true', default=False)
    args = parser.parse_args()

    # load hf 
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, truncation_side="left")
    # load corpus if needed
    if args.canonical != 0:
        collections = load_collections(args.collection)

    with open(args.topics, 'r') as fin, open(args.output, 'w') as fout:
        for line in tqdm(fin):
            data_dict = json.loads(line.strip())

            # extract data dict information
            topic_turn_id = data_dict['id']
            utterance = data_dict['utterance']
            history_utterances = data_dict['history_utterances']

            # prepare context
            if args.canonical != 0:
                n_history = len(history_utterances)
                n_selected = min(3, n_history)
                # laod responses
                history_responses = []
                for passage_id in data_dict['history_responses']:
                    history_responses.append(collections[passage_id])
                context = [f"{u} ||| {r}" for u, r in \
                        zip(history_utterances[-n_selected:], history_responses[-n_selected:])]
            else:
                # n_history = len(history_utterances)
                # n_selected = min(3, n_history)
                context = [f"{u}" for u in history_utterances]

            if args.add_topic:
                context = " ||| ".join([data_dict['topic']] + context + [utterance])
            else:
                context = " ||| ".join(context + [utterance])

            query_rewritten = rewrite(context)

            fout.write(f"{topic_turn_id}\t{query_rewritten}\n")


