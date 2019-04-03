""" Evaluate the baselines ont ROUGE"""
import argparse
import json
import os
from os.path import join, exists
from evaluate import eval_rouge


def main(args):
    dec_dir = join(args.decode_dir, 'output')
    ref_dir = args.reference_dir
    assert exists(ref_dir)

    dec_pattern = r'(\d+).dec'
    ref_pattern = '#ID#.ref'
    output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
    metric = 'rouge'
    print(output)
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    parser.add_argument('--reference_dir', action='store', required=True, help='directory of reference summaries')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)
