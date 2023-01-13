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

## Step2-constrat view pseudo labeling

1. Using the proposed method to generate the pseudo conversational query passage pairs
```
TBD
```

## Evaluation 

1. Download CAsT 2020
```
mkdir cast2020
```
2. Download ORConvQA dataset (dev set only)
```
mkdir orconqa
Download link: https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt
Download link: https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz
```
5. 
```
python3 tools/parse_canard.py 
```



## Note
To admit the google account in the local boto and gcloud authentication.

```
remove the old boto folder, which contained the previous credecential
rm ~/.boto

# new a config of glcoud by the commands

```

