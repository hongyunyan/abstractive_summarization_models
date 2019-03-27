""" decoding utilities"""
import json
import re
import os
from os.path import join
import pickle as pkl
from itertools import starmap

from cytoolz import curry

import torch

from hierarchical_model import HierarchicalSumm

from utils import PAD, UNK, START, END
from batcher import conver2id, pad_batch_tensorize
from data import CnnDmDataset


class DecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, data_path, split):
        assert split in ['val', 'test']
        super().__init__(split, data_path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0]))
    )['state_dict']
    return ckpt


class Abstractor(object):
    def __init__(self, abs_dir, max_sents=10, max_words=200, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        abs_args["embedding"] = None
        abs_ckpt = load_best_ckpt(abs_dir)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        abstractor = HierarchicalSumm(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_words = max_words
        self._max_sents = max_sents

    def _prepro(self, raw_article_sents):

        articles = conver2id(UNK, self._word2id, raw_article_sents)
        articles.sort(key=lambda data:len(data), reverse=True)
        article_lens = [len(art) for art in articles]

        art_sents = []
        for art in articles:
            for sent in art:
                art_sents.append(sent)
        
        sent_lens = [len(sent) for sent in art_sents]

        art_sents = pad_batch_tensorize(art_sents, PAD, cuda=False
                                     ).to(self._device)
        
        #art_sents 是一个每个文章所有sents凑在一起的矩阵，sent_lens是每个句子的长度，article_lens是每个文章有多少个句子
        dec_args = (art_sents, article_lens, sent_lens, START, END, self._max_sents, self._max_words)
        return dec_args

    def __call__(self, raw_article_sents):
        self._net.eval()
        dec_args = self._prepro(raw_article_sents)
        decs = self._net.batch_decode(*dec_args)

        dec_sents = []
        for i, raw_words in enumerate(raw_article_sents):
            dec = []
            for sent in decs[i]:
                for id_ in sent:
                    if id_ == END:
                        continue
                    else:
                        dec.append(self._id2word[id_])
            dec_sents.append(dec)
        return dec_sents


class BeamAbstractor(Abstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word),
                                 zip(all_beams, raw_article_sents)))
        return all_beams

@curry
def _process_beam(id2word, beam, art_sent):
    def process_hyp(hyp):
        seq = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == UNK:
                copy_word = art_sent[max(range(len(art_sent)),
                                         key=lambda j: attn[j].item())]
                seq.append(copy_word)
            else:
                seq.append(id2word[i])
        hyp.sequence = seq
        del hyp.hists
        del hyp.attns
        return hyp
    return list(map(process_hyp, beam))

class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        return article
