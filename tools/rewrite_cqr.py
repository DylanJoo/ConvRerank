from tqdm import tqdm
import argparse
import time
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer

def rewrite(texts):
    """
    Function of rewrite user utterance by inputting the context (utterances and responses),
    To reformulate the current user utterance.
    """
    input_ids = tokenizer(texts, return_tensors="pt", truncation=True).input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=512, num_beams=5)
    # rewrite_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True,)
    rewrite_text = tokenizer.decode(output_ids[0], skip_special_tokens=True,)
    return rewrite_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--model_name", type=str, default='castorini/t5-base-canard')
    parser.add_argument("--canonical", type=int, default=0)
    args = parser.parse_args()

    # load hf 
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, truncation_side="left")

    with open(args.topics, 'r') as fin, open(args.output, 'w') as fout:
        for line in tqdm(fin):
            data_dict = json.loads(line.strip())

            # extract data dict information
            topic_turn_id = data_dict['id']
            utterance = data_dict['utterance']
            history_utterances = data_dict['history_utterances']
            history_responses = data_dict['history_responses']

            # prepare context
            if args.canonical != 0:
                n_history = len(history_utterances)
                n_selected = min(3, n_history)
                context = [f"{u} ||| {r}" for u, r in \
                        zip(history_utterances[-n_selected:], history_responses[-n_selected:])]
            else:
                # n_history = len(history_utterances)
                # n_selected = min(3, n_history)
                context = [f"{u}" for u in history_utterances]

            context = " ||| ".join(context + [utterance])
            query_rewritten = rewrite(context)

            fout.write(f"{topic_turn_id}\t{query_rewritten}\n")


