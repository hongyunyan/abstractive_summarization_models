import math
import sys
import torch
from torch.nn import functional as F
import time

""" utility functions"""
import re
import os
from os.path import basename

import gensim
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence

def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3
EOA = 4 #end of article
#返回一个dict,key为word,value为序号
def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    word2id['<eoa>'] = EOA
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 5):  #most_common 返回一个list, list包含每个word和出现次数对 ,enumerate的第二个参数用来明确start iterator for index
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


#################### general sequence helper #########################
def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    #有种在填充有无字的感觉
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)  #(34,81),有字的地方为1，没有的为0
    return mask

def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean

def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))

    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)

    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


#################### LSTM helper #########################

def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states

#################### 转shape helper #################

def change_shape(input, input_lens, pad):
    #input为一个二维的矩阵， input_lens为一个list，里面写这应该新的output每行应该有几个input的行数

    batch_size = torch.tensor(input_lens)
    input_packed_sequence = PackedSequence(input, batch_size)

    output_pad_sequence, _ = pad_packed_sequence(input_packed_sequence, padding_value = pad)

    return output_pad_sequence

def change_reshape(input, input_lens):
    #三维矩阵转为二维矩阵
    output = [None] * len(input)
    # print(time.time()) 问题太多了，还有不按序列的问题，到时候测出来如果是bottle neck再考虑改
    # sort_ind = sorted(range(len(input_lens)),
    #                  key=lambda i: input_lens[i], reverse=True)
    # input_lens_new = [input_lens[i] for i in sort_ind] #再度排序
    # input_new = reorder_sequence(input, sort_ind)

    # input_lens_trans = []
    # for i in range(max(input_lens)):
    #     input_lens_trans.append(sum(j > i for j in input_lens))
    # output = [None] * len(input)
    # for index in range(len(input)): 
    #     output[index] = pack_padded_sequence(input[index].permute(1,0,2).transpose(0,1), input_lens_trans).data

    # print(time.time())
    for i in range(len(input_lens)):
        for j in range(input_lens[i]):
            if (i==0 and j==0):
                for index in range(len(input)):
                    output[index] = torch.unsqueeze(input[index][i][j], 0)
            else:
                for index in range(len(input)):
                    output[index] = torch.cat((output[index], torch.unsqueeze(input[index][i][j], 0)), dim=0)

    return output

def change_reshape_decoder(input, input_lens):
    #三维矩阵转为二维矩阵
    output = [None] * len(input)

    for i in range(len(input_lens)):
        for j in range(input_lens[i]):
            if (i==0 and j==0):
                for index in range(len(input)):
                    output[index] = torch.unsqueeze(input[index][j][i], 0)
            else:
                for index in range(len(input)):
                    output[index] = torch.cat((output[index], torch.unsqueeze(input[index][j][i], 0)), dim=0)

    return output