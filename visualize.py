import os
import argparse
from utils.viz import Viz
from utils.data_utils import load_npz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='submission.npz', type=str)
    args = parser.parse_args()

    viz = Viz('./data/config.json')
    viz.npz_to_images(load_npz(args.submission))
