# ConversationaPassageReranking (ConvRerank)

1. Requirements

Download Canard & QuAC
```
mkdir data
mkdir data/canard
mkdir data/quac

wget https://github.com/aagohary/canard/raw/master/data/release/train.json -P data/canard
wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json -P data/quac
wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json -P data/quac
```

Parse the QuAC and CANARD (some bug in datasset, I wrote a fixing codes)
```
python tools/parse_canard.py \
    -canard data/canard/train.json \
    -output data/canard.train.tsv \
    -quac data/quac
```
You will need a large-enough corpus to answer these questions (e.g. WIKI). In this repo, I used CAST2020 for demonstration (i.e. MSMARCO, TRECCAR).

2. Get sparse retrieval results of (Teacher) and (Student)

Used standard pyserini settings for top1000 relevant passages.
```
bash run_spr.sh answer+rewrite 
bash run_spr.sh rewrite

# The $query_type can be other value that existed in canard.train.csv
bash run_spr.sh utterance
```

3. Rerank the top1000 relevant passage 

To construct the denoising dataset, we use monot5 (castorini)
```
bash run_pre_rerank.sh
```

4. Constructed convir dataset
