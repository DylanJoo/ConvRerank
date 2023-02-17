import json
from tqdm import tqdm
import argparse
# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from utils import load_collections

def summarize(texts, args):
    # texts = f"summarize: {texts}"
    input_ids = tokenizer(texts, return_tensors="pt", truncation=True).input_ids.to(model.device)
    output_ids = model.generate(
            input_ids=input_ids,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            length_penalty=2.0,
            min_length=args.seq_length[0],
            max_length=args.seq_length[1],
            num_beams=args.beam_size,
    )
    summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True,)
    return summary_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--seq_length", type=int, nargs='+')
    parser.add_argument("--beam_size", type=int, default=5)
    args = parser.parse_args()

    # load hf 
    # model = T5ForConditionalGeneration.from_pretrained('t5-base')
    # tokenizer = T5Tokenizer.from_pretrained('t5-base')
    # model.cuda()
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model.cuda()

    # extract data dict information
    collections = load_collections(dir=args.collection)

    # load corpus if needed
    with open(args.topics, 'r') as fin, open(args.output, 'w') as fout:
        for line in tqdm(fin):
            data_dict = json.loads(line.strip())

            # collecting summary of each appeared canonical responses
            summarized_responses = []
            for passage_id in data_dict['history_responses']:
                if passage_id not in summarized_responses:
                    response_summary = summarize(collections[passage_id], args)
                    fout.write(f"{passage_id}\t{response_summary}\n")

                    summarized_responses.append(passage_id)


