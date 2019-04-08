""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from batcher import tokenize

from decoding import Abstractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe


def decode(save_path, model_dir, data_path, split, batch_size,
           beam_size, diverse, max_sents, max_words, cuda):
    start = time()

    if beam_size == 1:
        abstractor = Abstractor(model_dir, max_sents, max_words, cuda)
    else:
        abstractor = BeamAbstractor(model_dir, max_sents, max_words, cuda)

    # setup loader
    def coll(batch):
        #返回batch size的list，每个list中都是每篇文章每句话的list
        articles = list(filter(bool, batch))
        return articles

    dataset = DecodeDataset(data_path, split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )
    
    os.makedirs(join(save_path, 'output'))

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = tokenize(1500, raw_article_batch)
            #三层的list，最外层为article，下一层为sentence，再下一层为word
 
            if beam_size > 1:
                #先还没有实现
                all_beams = abstractor(tokenized_article_batch, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(tokenized_article_batch)
            # assert i == batch_size*i_debug

            for dec_out in dec_outs:
                decoded_sents = ['\n'.join(dec_out)]
                with open(join(save_path, 'output/{}.dec'.format(i)),'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print(decoded_sents)
            # for j, n in ext_inds:
            #     decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
            #     with open(join(save_path, 'output/{}.dec'.format(i)),
            #               'w') as f:
            #         f.write(make_html_safe('\n'.join(decoded_sents)))
            #     i += 1
            #     print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
            #         i, n_data, i/n_data*100,
            #         timedelta(seconds=int(time()-start))
            #     ), end='')
    print()

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--data_path', required=True, help='data_path')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')
    parser.add_argument('--max_dec_sent', type=int, action='store', default=10,
                        help='maximun sent to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir, args.data_path,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_sent, args.max_dec_word, args.cuda)
