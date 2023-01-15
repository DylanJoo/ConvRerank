# Weakly-supervised training data

## Step1-rerank
In this step, we need to generate three views of ranked list based on 
0. Manually reformulated query
1. Manually reformulated query + ANswer
x. Utterance

### Raw training data 
1. Download canard dataset (training set)
```
mkdir canard
Download link: https://sites.google.com/view/qanta/projects/canard
```

2. Download canard dataset (training set and validation set)
```
mkdir quac
Downlaod link: https://s3.amazonaws.com/my89public/quac/train_v0.2.json
Downlaod link: https://s3.amazonaws.com/my89public/quac/val_v0.2.json
```

3. Parsed and preprocess (and fix) the canard training set into data/canard/train.jsonl.
```
python3 tools/parse_canard.py 
```

4. Generated the first-stage ranked list using BM25 search
```
bash run_spr.sh
```

### Convert the runs into monot5 input
Converiting the training triplet into
```
# Input format: Query: <q> Docuemnt: <d> Relevant: 
# Output format: true/false
bash prepare_ranking_sources.sh
```

### monoT5 reranking
Followed the monoT5 paper, you can either using huggingface or GCP+TPU to get the results. 
Constructed the environments (See detail in [T5-for-IR](#))

```
mkdir monot5-probs
bash fetch_probs.sh
```

## Prposed pseudo labeling (conversational hard negative) and train on TPU

1. Using the proposed method to generate the pseudo conversational query passage pairs
```
mkdir data/canard4ir
# run the 
bash run_create_convir_dataset.sh
# or 
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections <corpus path>
  --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
  --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.txt \
  --window_size 3 \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \
```
2. Training with the weakly-supervsied dataset using TPU or GPU

```
TBD
```

## Evaluation on CAsT 2020
1. Download CAsT 2020
You may find out the download instructions (with preprocessing) in the [official CAsT 2020 repo](#).
Then, parse the evaluation topics into jsonl
- Qrels
- Topics
- Collections (MARCO, TRECCR)
```
python3 tools/parse_cast2020.py
```
2. First-stage retrieval using CQE
We have inferenced several dense retrieval baseline in this repo, including the top1000 passage ranklist which are in [cast2020 runs](runs/cast2020/)
3. Convert the runs into monot5 input
```
# run the bash script
bash prepare_cast2020_evaluation.sh
# or 
python3 tools/convert_runs_to_monot5.py \
  --run runs/cast2020.eval.cqe.trec \
  --topic data/canard/train.jsonl \
  --collection <corpus path> \
  --output monot5/cast2020.eval.cqe.rerank.txt 
```
4. Predicted the relevance scores using fine-tuned t5. You can see our checkpoint at [bucket](#).

## Evaluation on ORConvQA
1. Download ORConvQA dataset (dev set only)
```
TBD
```

## Note
To admit the google account in the local boto and gcloud authentication.

```
remove the old boto folder, which contained the previous credecential
rm ~/.boto

# new a config of glcoud by the commands

```

