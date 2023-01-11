import os
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneSearcher

def search(args):
    # Lucuene initialization
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # Load query text and query ids
    qid = []
    qtext = []
    with open(args.topics) as f:
        for line in f:
            data = json.loads(line.strip())
            qvalues = ""
            for value in args.query.split("+"):
                qvalues += " "+ data[value]
            qid.append(data['id'])
            qtext.append(qvalues.strip())

    # Prepare the output file
    output = open(args.output, 'w')

    # search for each q
    for index, text in tqdm(zip(qid, qtext)):
        hits = searcher.search(text, k=args.k)
        for i in range(len(hits)):
            output.write(f'{index} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} pyserini\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--k1",type=float)
    parser.add_argument("--b", type=float)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--output", default='run.sample.txt', type=str)
    # special args
    parser.add_argument("--topics", default=None, type=str)
    parser.add_argument("--query", default='rewrite', type=str)
    args = parser.parse_args()

    os.makedirs('runs', exist_ok=True)
    search(args)
    print("Done")
