import collections
import pandas as pd
import argparse

def analyze_by_turn(args):
    
    with open(args.run0, 'r') as f:
        lines = [tuple(line.split('\t')) for line in f.readlines()]
        run0 = dict(lines)

    with open(args.runx, 'r') as f:
        lines = [tuple(line.split('\t')) for line in f.readlines()]
        runx = dict(lines)
    assert len(run0) == len(runx), 'inconsistent length'

    with open(args.runy, 'r') as f:
        lines = [tuple(line.split('\t')) for line in f.readlines()]
        runy = dict(lines)
    assert len(run0) == len(runy), 'inconsistent length'

    turn_results=[]
    for qid in run0.keys():
        if qid == 'all':
            break
        turn = int(qid.split('_')[-1])
        score0 = float(run0[qid])
        scorex = float(runx[qid])
        scorey = float(runy[qid])
        turn_results.append([
            turn, scorex-score0, scorey-score0
            # turn, int(scorex-score0>0), int(scorey-score0>0)
        ])

    df = pd.DataFrame(turn_results, columns=['turn', 'x-0 (monot5)', 'y-0 (convrerank)'])
    df = df.groupby('turn').agg(['mean', 'count'])
    print(df)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run0", type=str, default='run0.txt')
    parser.add_argument("--runx", type=str, default='runx.txt')
    parser.add_argument("--runy", type=str, default='runy.txt')
    parser.add_argument("--output", type=str, default="runx-runy.csv")
    args = parser.parse_args()

    df=analyze_by_turn(args)
    df.to_csv(args.output)
