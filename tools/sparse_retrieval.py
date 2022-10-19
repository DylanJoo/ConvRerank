import json
import argparse
from pyserini.search.lucene import LuceneSearcher

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--retrieve_k_docs", default=1000, type=int)
parser.add_argument("-k1", "--k1", type=float)
parser.add_argument("-b", "--b", type=float)
parser.add_argument("-index", "--dir_index", default=None, type=str)
parser.add_argument("-qid", "--path_qid", default=None, type=str)
parser.add_argument("-query", "--path_qtext", default='sample_queries.tsv', type=str)
parser.add_argument("-qval", "--qvalue", type=str)
parser.add_argument("-output", "--path_output", default='run.sample.txt', type=str)
args = parser.parse_args()

def search(args):
    # Lucuene initialization
    searcher = LuceneSearcher(args.dir_index)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # Load query text and query ids
    if args.path_qid:
        query_text = open(args.path_qtext, 'r').read().splitlines()
        query_id = open(args.path_qid, 'r').read().splitlines()
        query_file = zip(query_id, query_text)
    elif "tsv" in args.path_qtext:
        query_file = [(i.split('\t')) for i in open(args.path_qtext, 'r').read().splitlines()]
    elif "json" in args.path_qtext:
        query_file = []
        with open(args.path_qtext) as f:
            for line in f:
                data = json.loads(line.strip())
                if "+" in args.qvalue:
                    qvalues = ""
                    for qvalue in args.qvalue.split("+"):
                        qvalues += " "+data[qvalue]
                    query_file.append((data['id'], qvalues))
                else:
                    query_file.append((data['id'], data[args.qvalue]))

    # Prepare the output file
    output = open(args.path_output, 'w')

    # search for each q
    for qi, (index, text) in enumerate(query_file):
        hits = searcher.search(text.strip(), k=args.retrieve_k_docs)
        for i in range(len(hits)):
            output.write(f'{index} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} pyserini\n')
        if qi % 100 == 0:
            print(f'{qi+1} query retrieved ...')

search(args)
print("Done")
