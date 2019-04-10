""" train the abstractor"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from data import CnnDmDataset
from batcher import coll_fn, prepro_fn
from batcher import convert_batch, batchify_fn
from batcher import BucketedGenerater

from utils import sequence_loss
from utils import PAD, UNK, START, END, EOA
from utils import make_vocab, make_embedding

from hierarchical_model import HierarchicalSumm


import time

BUCKET_SIZE = 6400


class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, args.data_path)

    def __getitem__(self, i):
        #改为返回source为原文,target为abstract,然后两个都是list？
        js_data = super().__getitem__(i)
        art_sents, abs_sents = (
            js_data['artile'], js_data['abstract'])
        return art_sents, abs_sents


def configure_net(vocab_size, emb_dim,
                  n_hidden, bidirectional, n_layer, sampling_teaching_force, embedding):
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['n_hidden']      = n_hidden
    net_args['bidirectional'] = bidirectional
    net_args['n_layer']       = n_layer
    net_args['embedding']     = embedding
    net_args['sampling_teaching_force'] = sampling_teaching_force

    net = HierarchicalSumm(**net_args)
    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    
    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    return criterion, train_params

def build_batchers(word2id, cuda, debug):
    prepro = prepro_fn(args.max_word)
    batchify = compose(
        batchify_fn(PAD, START, END, EOA, cuda=cuda),
        convert_batch(UNK, word2id)
    )  #这玩意竟然是倒着开始执行的？？？？？？

    train_loader = DataLoader(
        MatchDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        MatchDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher

def main(args):
    # create data batcher, vocabulary
    # batcher
    with open(join(args.data_path, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize, args.max_target_sent) #一个word的词典
    train_batcher, val_batcher = build_batchers(word2id, args.cuda, args.debug)

    # make net
    
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            {i: w for w, i in word2id.items()}, args.w2v) #提供一个embedding矩阵

        net, net_args = configure_net(len(word2id), args.emb_dim,
                                  args.n_hidden, args.bi, args.n_layer, args.sampling_teaching_force, embedding)
    else:
        print("please provide pretrain_w2v")
        return 

    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    
    net_args_backup = net_args.copy()
    del net_args_backup["embedding"]

    meta = {}
    meta['net']           = 'base_abstractor'
    meta['net_args']      = net_args_backup
    meta['traing_params'] = train_params

    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)


    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )

    parser.add_argument('--path', required=True, help='root of the model')

    parser.add_argument('--data_path', required=True, help='root of the dataset')

    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_target_sent', type=int, action='store', default=10,
                        help='maximum sentence num in the summary')
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=16,
                        help='the training batch size')
    parser.add_argument('--sampling_teaching_force', type=bool, action='store', default=False,
                        help='choose whether use scheduled sampling for teaching force algorithm')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=10000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=4,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    print(args)

    main(args)
