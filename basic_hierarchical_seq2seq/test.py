from decoding import load_best_ckpt
from hierarchical_model import PretrainModel
import pickle as pkl
from batcher import conver2id,pad_batch_tensorize
from utils import UNK,PAD,START,END
import os
from os.path import join
from utils import make_vocab, make_embedding
import torch


word2id = pkl.load(open(join("abstractor", 'vocab.pkl'), 'rb'))
embedding, _  = make_embedding({i: w for w, i in word2id.items()}, "word2vec/word2vec.128d.226k.bin")
id2word = {i: w for w, i in word2id.items()}

net_args = {}
net_args['vocab_size']    = 50025
net_args['emb_dim']       = 128
net_args['n_hidden']      = 256
net_args['bidirectional'] = True
net_args['n_layer']       = 1
net_args['sampling']       = False
net_args["embedding"] = None

ckpt = load_best_ckpt("pretrain_abstractor_0515")

abstractor = PretrainModel(**net_args)
abstractor.load_state_dict(ckpt, strict=False)
abstractor = abstractor.cuda()

# sent = [[['student', 'is', 'no',"longer", "on", "duke", "university", "campus", "and", "will", "face", "disciplinary", "review", "."]]]
articles  = [[[5]]]
# articles = conver2id(UNK, word2id, sent)
# print(articles)
length = len(articles[0][0])
tar_in = [[[[START] + articles[0][0]]]]
target = [[[articles[0][0] + [END]]]]

article = torch.cuda.LongTensor(1,length)
article[0, :] = torch.cuda.LongTensor(articles[0][0])

tar_input = torch.cuda.LongTensor(1,length+1)
target_output = torch.cuda.LongTensor(1,length+1)
tar_input[0, :] = torch.cuda.LongTensor(tar_in[0][0])
target_output[0, :] = torch.cuda.LongTensor(target[0][0])

article_lens = [length]
target_length = [length+1]
output  = abstractor.decode(article, article_lens)
print(output)
# for i in output:
#     print(id2word[i])

logit = abstractor(article, article_lens, tar_input, target_length)
print(logit, logit.size())

for i in range(length+1):
    item = target[0][0][0][i]
    print(item)
    print(logit[0][i][item])


tok = torch.max(logit[0],dim=1)[1]
pro = torch.max(logit[0],dim=1)[0]
print(tok, ",", pro)
print(pro.size())




