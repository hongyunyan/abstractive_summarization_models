import torch
from time import time
from math import pow
import random
from torch import nn
from torch.nn import init

from rnn import lstm_encoder
from rnn import MultiLayerLSTMCells
from attention import step_attention
from utils import sequence_mean, len_mask, change_shape, change_reshape, change_reshape_decoder, change_loss_shape

from torch.nn import functional as F
from utils import reorder_sequence, reorder_lstm_states

from utils import EOA, PAD, special_word_num
from seq2seq import Seq2SeqSumm
import beamsearch as bs

from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
import torch.multiprocessing as mp
from beamsearch import Beam

from sub_model import WordToSentLSTM, HierarchicalWordToSentLSTM, SentToWordLSTM

INIT = 1e-2

class PretrainModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_hidden, bidirectional, n_layer, embedding, dropout=0.0):

        super().__init__()

        self._bidirectional = bidirectional
        # enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(n_hidden, emb_dim, bias=False)

        self._n_hidden = n_hidden

        self._WordToSentLSTM = WordToSentLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, None, embedding)
        self._SentToWordLSTM = SentToWordLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, None, embedding)

    def forward(self, source_sents, source_length, tar_inputs, target_length):
        
        sent_output = self._WordToSentLSTM(source_sents, source_length)  
        context_output = self._dec_h(sent_output)
        logit = self._SentToWordLSTM(context_output, tar_inputs, None, None)

        return logit


class HierarchicalSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, sampling_teaching_force, self_attn, hi_encoder, embedding, dropout=0.0):
        super().__init__()

        self._bidirectional = bidirectional
        self._hidden_output = nn.Linear(emb_dim, n_hidden, bias=False)
        self._n_hidden = n_hidden
        self._self_attn = self_attn
        self._hi_encoder = hi_encoder
        self._Seq2SeqSumm = Seq2SeqSumm(vocab_size, n_hidden, n_hidden, bidirectional, n_layer, dropout, embedding)
        self._WordToSentLSTM = WordToSentLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, self_attn, embedding)
        self._SentToWordLSTM = SentToWordLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, sampling_teaching_force, embedding)
        self._HierarchicalWordToSentLSTM = HierarchicalWordToSentLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, self_attn, embedding)
    
    def forward(self, article_sents, article_lens, sent_lens,  abstract_sents, abs_lens):  
        #传给第一个函数的article_sents是一个n个句子×m个word， sent_lens为每个句子的长度的n维矩阵，矩阵article_lens，记录每个article有几个句子

        #尝试一个多层的wordtosent，天哪我在干什么orz
        wordToSentModel = self._HierarchicalWordToSentLSTM if self._hi_encoder else self._WordToSentLSTM
        sent_vec = wordToSentModel(article_sents, sent_lens)  #[batch, 256]，每个vec表示的是每句话的信息

        pad = 1e-8  #用来填充没有句子的地方的hidden        
        #根据上面那个[batch，256]，改道成一个文章数×句子长×256的矩阵 article_hidden_states !!!
        article_hidden_states = change_shape(sent_vec, article_lens, pad)
        
        #然后句子的输入开始过最基本的seq2seq层
        (sent_dec_out, sent_h_out, sent_c_out), loss_part_all = self._Seq2SeqSumm(article_hidden_states, article_lens, abs_lens)
        #感觉这边要设置一下，如何让sent数目输出的是正确的？？？？加入loss？？？？
        #坑坑坑来来来转格式了，从文章数×target句子长×256 到所有句子数×256  sentence_hidden_states!!
        sent_output = change_reshape([sent_dec_out, sent_h_out, sent_c_out], abs_lens)
        sentence_output_states, sentence_hidden_states, sentence_context_states= sent_output[:]

        loss_part_output = change_loss_shape(loss_part_all, abs_lens)
        loss_part = torch.mean(loss_part_output)
        #获得句子的每个hidden以后，一生多 生成每个句子, 然后每个生成的具体句子跟原始的target做loss，返回loss
        logit = self._SentToWordLSTM(sentence_output_states, abstract_sents, sentence_hidden_states, sentence_context_states)

        return logit, loss_part
    
    
    def batch_decode(self, article_sents, article_lens, sent_lens, start, max_sent, max_words):
        """ greedy decode support batching"""
        
        sent_vec = self._WordToSentLSTM(article_sents, sent_lens)  

        pad = 1e-8  #用来填充没有句子的地方的hidden
        article_hidden_states = change_shape(sent_vec, article_lens, pad)

        #这边要改成encoder和decoder分离，先让已知的数据穿过seq2seq的encoder然后获得参数开始decoder 句子vec，然后再用句子vec decoder出sent vec
        #或者这边应该调用seq2seq的beam_decoder函数，让他自己decoder好返回给我下一步decoder的参数
        sent_dec_out, sent_h_out, sent_c_out = self._Seq2SeqSumm.batch_decode(article_hidden_states, article_lens, max_sent)

        #快乐继续换格式
        sent_output = change_reshape_decoder([sent_dec_out, sent_h_out, sent_c_out])
        sentence_output_states, sentence_hidden_states, sentence_context_states = sent_output[:]

        init_states = (torch.unsqueeze(sentence_hidden_states, 0).contiguous(),
                       torch.unsqueeze(sentence_context_states, 0).contiguous())

        states = init_states

        tok = torch.cat([torch.arange(max_sent) + special_word_num] * len(article_lens), dim=0).to(article_sents.device)
        dec_out = sentence_output_states
        outputs = None
        pre_tok_2 = (torch.ones(tok.size()[0],1) * -1).long().cuda()
        pre_tok_1 = (torch.ones(tok.size()[0],1) * -1).long().cuda()
        for i in range(max_words):
            logit, states, dec_out = self._SentToWordLSTM._step(tok, states, dec_out)
            logit[:, special_word_num: max_sent + special_word_num] = -10000 #刨除可能出现的sent_
            tok = torch.max(logit, dim=1, keepdim=True)[1]  #挣扎一下维度对不对
            for index in range(tok.size()[0]):
                if (tok[index,0] == pre_tok_1[index, 0] and tok[index,0] == pre_tok_2[index, 0]):
                    logit[index, tok] = -10000 
            tok = torch.max(logit, dim=1, keepdim=True)[1]
            pre_tok_2 = pre_tok_1 
            pre_tok_1 = tok

            if (i == 0): 
                outputs = torch.unsqueeze(tok[:,0] , 1)
            else:
                outputs = torch.cat((outputs, torch.unsqueeze(tok[:, 0] , 1)), dim=1)

        #分成文章
        articles_output = []
        sent_num = 0
        for i in range(len(article_lens)):
            sents = outputs[sent_num: sent_num + max_sent]
            sent_num = sent_num + max_sent
            article = []
            for sent in sents:
                sent_ids = []
                for word in sent:
                    sent_ids.append(word.item())
                article.append(sent_ids)
            articles_output.append(article)
        return articles_output

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def batched_beam_search(self, article_sents, article_lens, sent_lens,
                            start, max_sent, max_words, eos, beam_size, diverse=1.0):

        sent_vec = self._WordToSentLSTM(article_sents, sent_lens)  

        pad = 1e-8  #用来填充没有句子的地方的hidden
        article_hidden_states = change_shape(sent_vec, article_lens, pad)

        #这边要改成encoder和decoder分离，先让已知的数据穿过seq2seq的encoder然后获得参数开始decoder 句子vec，然后再用句子vec decoder出sent vec
        #或者这边应该调用seq2seq的beam_decoder函数，让他自己decoder好返回给我下一步decoder的参数
        sent_dec_out, sent_h_out, sent_c_out = self._Seq2SeqSumm.batch_decode(article_hidden_states, article_lens, max_sent)

        #快乐继续换格式
        sent_output = change_reshape_decoder([sent_dec_out, sent_h_out, sent_c_out])
        sentence_output_states, sentence_hidden_states, sentence_context_states = sent_output[:]

        article_num = len(article_lens)
        batch_size = sentence_output_states.size()[0]


        #这边开始糊了一版beam_search，大致思想是我永远更新一个([beam_sent*beam_size]*batch_size)的一个list称为beams;最终目的是获得一个results lists，results中每个list有beam_size个完整的句子
        #首先在每个step，我先把beams中的所有内容变成一个矩阵，让他经过step去生成下一个词，获得新的context，states啊bla的
        #然后我就更新每句话对应的result和beams，
        beams = [[Beam(tokens=[i%max_sent + special_word_num],
                      log_probs=[0.0],
                      state=(sentence_hidden_states[i], sentence_context_states[i]),
                      context = sentence_output_states[i]) for j in range(beam_size)]
                for i in range(batch_size)] #新建batch_size的beams,外层batch_size个，内层beam_size个

        results = []
        for i in range(batch_size):
            results.append([])
        
        for step in range(max_words):
            stop_flag = True
            latest_tokens = [sent.latest_token for item_beam in beams for sent in item_beam]
            tok = torch.tensor(latest_tokens).cuda()

            all_state_h = []
            all_state_c = []
            all_context = []
            for beam_item in beams:
                for sent in beam_item:
                    h, c = sent.state
                    all_state_c.append(c)
                    all_state_h.append(h)
                    all_context.append(sent.context)

            states = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            context = torch.stack(all_context, 0)

            logit, states = self._SentToWordLSTM._step(tok, states, context)
            topk_log_probs, topk_ids = logit.topk(beam_size) #确认一下dim是否正确

            all_beams = []
            for i in range(batch_size):
                all_beams.append([])

            for i in range(batch_size):
                for j in range(beam_size):
                    sent = beams[i][j]
                    state_i = (states[0][0][i*beam_size+j], states[1][0][i*beam_size+j])
                    context_i = context[i*beam_size+j]

                    for k in range(beam_size):
                        new_beam = sent.extend(token=topk_ids[i*beam_size+j, k].item(),
                                        log_prob=topk_log_probs[i*beam_size+j, k].item(),
                                        state=state_i,
                                        context=context_i)
                        all_beams[i].append(new_beam)

            beams = []
            for i in range(batch_size):
                beams.append([])

            for i in range(batch_size):
                for sent in self.sort_beams(all_beams[i]):
                    if (sent.latest_token == EOA or sent.latest_token == eos):
                        results[i].append(sent)
                    else:
                        beams[i].append(sent)
                    if (len(beams[i]) == beam_size):
                        if (len(results[i]) < beam_size):
                            stop_flag = False 
                        break

                if (len(results[i]) > beam_size):
                    results[i] = self.sort_beams(results[i])[:beam_size]
            if (stop_flag == True):
                break
                
        best_sent = []
        for i in range(batch_size):
            if (len(results[i])==0):
                best_sent.append(None)
            else:
                best_sent.append(self.sort_beams(results[i])[0])
        
        
        articles_output = []
        for i in range(article_num):
            article = []
            for j in range(max_sent):
                if (best_sent[j] is None):
                    continue
                article.append(best_sent[j].tokens[1:-1])
                if best_sent[j].tokens[-1] == EOA:
                    break
            articles_output.append(article)
        return articles_output
                    
        
        










       

