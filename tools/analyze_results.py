import collections
import pandas as pd
import argparse

def analyze_by_turn(args):
    
    with open(args.result0, 'r') as f:
        lines = [tuple(line.split('\t')) for line in f.readlines()]
        result0 = dict(lines)

    with open(args.resultx, 'r') as f:
        lines = [tuple(line.split('\t')) for line in f.readlines()]
        resultx = dict(lines)
    assert len(result0) == len(resultx), f'inconsistent length, len(result0) and len(resultx)'

    with open(args.resulty, 'r') as f:
        lines = [tuple(line.split('\t')) for line in f.readlines()]
        resulty = dict(lines)
    assert len(result0) == len(resulty), f'inconsistent length, len(result0) and len(resulty)'

    turn_results=[]
    for qid in result0.keys():
        if qid == 'all':
            break
        turn = int(qid.split('_')[-1])
        score0 = float(result0[qid])
        scorex = float(resultx[qid])
        scorey = float(resulty[qid])
        if args.compare == 'wtl':
            a = int(scorex >= score0)
            b = int(scorey >= score0)
        elif args.compare == 'avg':
            a = scorex-score0
            b = scorey-score0
        else:
            a, b = 0, 0

        turn_results.append([qid, turn, a, b])

    df = pd.DataFrame(turn_results, columns=['qid', 'turn', f'monoT5 - {args.compare}', f'ConvRerank - {args.compare}'])
    df = df.set_index('qid')
    df.to_csv(f"{args.resultx}-{args.resulty}.csv")
    print(df.drop(columns='turn').sum(axis=0))
    if args.compare == 'wtl':
        print(df.groupby('turn').agg(['sum', 'count']))
        print(df.groupby('turn').agg(['sum'])[f'monoT5 - {args.compare}'].to_dict())
        print(df.groupby('turn').agg(['sum'])[f'ConvRerank - {args.compare}'].to_dict())
    elif args.compare == 'avg':
        print(df.groupby('turn').agg(['mean', 'count']))
        print(df.groupby('turn').agg(['mean'])[f'monoT5 - {args.compare}'].to_dict())
        print(df.groupby('turn').agg(['mean'])[f'ConvRerank - {args.compare}'].to_dict())
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--result0", type=str, default='result0.txt')
    parser.add_argument("--resultx", type=str, default='resultx.txt')
    parser.add_argument("--resulty", type=str, default='resulty.txt')
    parser.add_argument("--runx", type=str, default='runx.txt')
    parser.add_argument("--runy", type=str, default='runy.txt')
    parser.add_argument("--collections", type=str, default='/tmp2/jhju/datasets/cast2020/')
    parser.add_argument("--topics", type=str, default='data/cast2019/cast2019.eval.jsonl')
    parser.add_argument("--compare", type=str, default='wtl')
    parser.add_argument("--show_cases", action='store_true', default=False)
    parser.add_argument("--output", type=str, default="resultx-resulty.csv")
    args = parser.parse_args()

    df = analyze_by_turn(args)

    if args.show_cases:
        from utils import load_runs, load_collections, load_topics
        """
        Find out the greatest condition (of query) that ConvRerank outperform monoT5
        """
        runx = load_runs(args.runx)
        runy = load_runs(args.runy)
        queries = load_topics(args.topics)
        passages = load_collections(dir=args.collections)

        df['diff'] = df[f'ConvRerank - {args.compare}'] - df[f'monoT5 - {args.compare}']

        ## largest gap
        imax = df['diff'].argmax()
        qid=df.iloc[imax, :].name
        print(f'Question: {queries[qid]}\nmonoT5')
        for docid in runx[qid][:5]:
            print(docid)
            print(passages[docid])
        print('\nConvRerank')
        for docid in runy[qid][:5]:
            print(docid)
            print(passages[docid])

        ## largest gap with positive
        imaxs = df[(df[f'monoT5 - {args.compare}'] > 0) & (df[f'ConvRerank - {args.compare}'] > 0)].sort_values('diff', ascending=False)
        imaxs = imaxs[imaxs['diff'] > 0].index
        for qid in imaxs:
            print(f'Question: {queries[qid]}\nmonoT5')
            for docid in runx[qid][:5]:
                print(docid)
                print(passages[docid])
            print('\nConvRerank')
            for docid in runy[qid][:5]:
                print(docid)
                print(passages[docid])
