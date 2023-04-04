# Improving Conversational Passage Re-ranking with View Ensemble

This repositary includes
1. Step1-rerank: Retrieve and re-rank the passage with two views.
2. Proposed pseudo labeling: Create the pseudo-labels for CANARD dataset based on the view-ensemble ranked list.
3. Train conversational passage re-ranker
4. Evaluation on TREC CAsT 2019 and 2020.
---

## Step1-rerank
In this step, we need to generate three views of ranked list based on 

1. Manually reformulated query(CANARD)
2. Manually reformulated query(CANARD) + Answer (QuAC)

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
Constructed the environments, See detail in [T5-for-IR](#)
```
mkdir monot5-probs
bash fetch_probs.sh
```

## Proposed pseudo labeling (conversational hard positive/negative sampling) 
1. Using the proposed method to generate the pseudo conversational query passage pairs
Run the bash files (note that the other ablation datasets are also included)
```
mkdir data/canard4ir
bash run_create_convir_dataset.sh
# or 
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.jsonl \
  --collections <corpus path>
  --run0 runs/cast20.canard.train.view0.monot5.top1000.trec \
  --run1 runs/cast20.canard.train.view1.monot5.top1000.trec \
  --output data/canard4ir/canard4ir.train.convrerank.txt \
  --topk_pool 200 \
  --topk_positive 20 \
  --n 20 \
```
2. Training with the weakly-supervsied dataset using TPU or GPU
See the [T5_TPU_README.md](T5_TPU_README.md) for detail.

## Evaluation on CAsT 2020 (eval)
1. Download CAsT 2020
The evaluation files are processed and parsed from original CAsT'20 repositary, in [data/cast2020](data/cast2020/)
You can also download from the [official CAsT repo](#), and follow our processing pipeline.
2. First-stage retrieval using CQE
The dense retrieval results are in this repo, including the top1000 passage ranklist which are in [runs/cast2020](runs/cast2020/)
3. Convert the runs into monot5 input
```
python3 tools/convert_runs_to_monot5.py \
  --run <run file> \
  --topic data/cast2020/cast2020.eval.jsonl \ 
  --collection <corpus path> \
  --output monot5/cast2020.eval.cqe.conv.rerank.txt 
  --conversational
```
4. Predicted the relevance scores using fine-tuned t5. You can see our checkpoint at [Google bucket](#).
- monot5-large-canard4ir
- monot5-base-canard4ir

## Evaluation on CAsT 2019
1. Download CAsT 2019
You may find out the download files in [official CAsT repo](#)
Then, parse the evaluation topics into jsonl.
- Qrels (already in this repo)
- Collections (MARCO, TRECCAR, WAPO)
2. First-stage retrieval using CQE
We have inferenced several dense retrieval baseline in this repo, including the top1000 passage ranklist which are in [cast2019 runs](runs/cast2019/)
3. Convert the runs into monot5 input
```
python3 tools/convert_runs_to_monot5.py \
  --run <run file> \
  --topic data/cast2019/cast2019.eval.jsonl \ 
  --collection <corpus path> \
  --output monot5/cast2019.eval.cqe.conv.rerank.txt 
  --conversational
```
4. Predicted the relevance scores using fine-tuned t5. You can see our checkpoint at [bucket](#).
- monot5-base-canard4ir
- monot5-large-canard4ir

