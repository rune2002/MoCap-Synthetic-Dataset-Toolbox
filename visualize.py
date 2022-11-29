import os
import argparse
from utils.viz import Viz
from utils.data_utils import load_npz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='submission.npz', type=str)
    parser.add_argument('--output_dir', default='./visualize', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    viz = Viz('./data/config.json')
    viz.npz_to_images(load_npz(args.submission), args.output_dir)