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

2. Get sparse retrieval results of (Teacher) and (Student), which I used standard pyserini settingsfor top1000 relevant passages.

```
bash run_spr.sh answer+rewrite 
bash run_spr.sh rewrite

# The $query_type can be other value that existed in canard.train.csv
e.g.
bash run_spr.sh utterance
```
