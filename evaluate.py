import os
import argparse
from utils.scorer import Scorer
from utils.data_utils import load_npz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='submission.npz', type=str)
    parser.add_argument('--target', default='./data/target.npz', type=str)
    args = parser.parse_args()

    scorer = Scorer(load_npz(args.submission), load_npz(args.target))
    scorer.report()