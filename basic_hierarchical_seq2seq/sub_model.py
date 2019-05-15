import torch
from time import time
from math import pow
import random
from torch import nn
from torch.nn import init

from rnn import lstm_encoder
from rnn import MultiLayerLSTMCells
from attention import step_attention
from utils import sequence_mean, len_mask, change_shape, change_reshape, change_reshape_decoder

from torch.nn import functional as F
from utils import reorder_sequence, reorder_lstm_states

from utils import EOA, PAD, special_word_num
from seq2seq import Seq2SeqSumm
import beamsearch as bs

from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence

INIT = 1e-2

class WordToSentLSTM(nn.Module):
    def __init__(self, emb_dim, n_hidden, n_layer,
            bidirectional, dropout, vocab_size, self_attn, embedding):
        super().__init__()

        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding is not None:
            assert self._embedding.weight.size() == embedding.size()
            self._embedding.weight.data.copy_(embedding)
        self._self_attn = self_attn

        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        #用来加一层转换输出的格式
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)

        self._lstm_layer = nn.LSTM(input_size = emb_dim, hidden_size= n_hidden, num_layers = n_layer, bidirectional = bidirectional, dropout = dropout)

        if (self_attn == True and bidirectional == True):
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * n_hidden  ,2 * n_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * n_hidden))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2* n_hidden, 1))

            self.weight_W_sent.data.uniform_(-0.1, 0.1)
            self.weight_proj_sent.data.uniform_(-0.1,0.1)

    def lstm(self, sequence, seq_lens, init_enc_states, need_embedding = True):
        #输出为batch，hidden_dim
        batch_size = sequence.size(0)
        if (need_embedding):
            #注入embedding matrix
            sequence = sequence.transpose(0, 1)
            emb_sequence = self._embedding(sequence)
        else:
            emb_sequence = sequence.transpose(0,1)

        #再度排序
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)),
                        key=lambda i: seq_lens[i], reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind] 
        emb_sequence = reorder_sequence(emb_sequence, sort_ind)
        #扔进lstm
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence, seq_lens)  
        packed_out, final_states = self._lstm_layer(packed_seq, init_enc_states) 
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out) 

        #再把位置调回来
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind)
        final_states = reorder_lstm_states(final_states, reorder_ind)

        return lstm_out, final_states
    
    def self_attn(self, hidden_output):
        #self attention
        sent_origin = torch.tanh(torch.matmul(hidden_output, self.weight_W_sent) + self.bias_sent.expand(hidden_output.size())) #u_it = tanh(w_w*h_it+b)
        sent_attn = torch.matmul(sent_origin, self.weight_proj_sent)
        sent_attn_norm = F.softmax(sent_attn + 1e-8 ) + 1e-8 
        sent_output = torch.sum(torch.mul(sent_attn_norm.expand(sent_origin.size()), hidden_output), dim=1)

        return sent_output

    def forward(self, article_sents, sent_lens, need_embedding = True):
        size = (
            self._init_enc_h.size(0),
            len(sent_lens),
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        init_states = (init_enc_states[0].contiguous(),
                       init_enc_states[1].contiguous())  #传闻中变成连续块的函数

        lstm_out, final_states  = self.lstm(article_sents, sent_lens, init_states, need_embedding)
        
        if (self._self_attn):
            sent_output = self.self_attn(lstm_out.transpose(0,1))
            sent_output = torch.stack([self._dec_h(h) for h in sent_output], dim=0) #从 [batch,512] 到 [batch,256]

            return sent_output
        else:

            output = torch.cat(final_states[0].chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】
            output = torch.stack([self._dec_h(h) for h in output], dim=0)[-1] #从 [1,batch,512] 到 [batch,256]

            return output


class HierarchicalWordToSentLSTM(WordToSentLSTM):
    def __init__(self, emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, self_attn, embedding):
        super().__init__(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, self_attn, embedding)

        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self.dec_h = nn.Linear(enc_out_dim, emb_dim, bias=False)
        self.bidirectional = bidirectional

    
    def devide(self, article_sents, sent_lens, divide_num=8):
        #先扩充成divide_num的倍数
        pad_num = ((article_sents.size()[1]//divide_num + 1)*divide_num - article_sents.size()[1]) % divide_num #要给原有的article_sents填充几个pad

        article_sents_with_pad = torch.cat([article_sents, torch.tensor(PAD).expand(article_sents.size()[0], pad_num)], dim=1) #句子数×8n
        sent_lens = [lens // divide_num + (lens % divide_num > 0) for lens in sent_lens] 
        article_sents_reshape = article_sents_with_pad.reshape(-1, divide_num)
 
        mask = torch.tensor([sent[0]!=0 for sent in article_sents_reshape]) 
        article_sents_deivide = torch.masked_select(article_sents_reshape, mask.expand(divide_num, mask.size()[0]).transpose(0,1)).reshape(-1, divide_num)

        article_sents_deivide_lens = torch.sum(article_sents_deivide > 0, 1)
        return article_sents_deivide, article_sents_deivide_lens, sent_lens


    def forward(self, article_sents, sent_lens):
        #排序排序，按长度倒序倒序

        sort_ind = sorted(range(len(sent_lens)),
                        key=lambda i: sent_lens[i], reverse=True)
        sent_lens = [sent_lens[i] for i in sort_ind] 
        article_sents = reorder_sequence(article_sents, sort_ind, True)

        #首先divide 8 分句子
        article_sents_divided, article_sents_deivide_lens, sent_divided_lens = self.devide(article_sents, sent_lens)

        #扔去做第一层
        output = super().forward(article_sents_divided, article_sents_deivide_lens)

        if (self._self_attn):
            output = torch.stack([self.dec_h(h) for h in output], dim=0) #从 [batch,512] 到 [batch,128]
        else:
            if (self.bidirectional):
                output = torch.cat(output.chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】
            output = torch.stack([self.dec_h(h) for h in output], dim=0)[-1] #从 [1,batch,512] 到 [batch,128]

        #套用楼上得到结果
        #搞第二层，先改个格式
        sent_divided_lens_tensor = torch.tensor(sent_divided_lens)
        input_packed_sequence = PackedSequence(output, sent_divided_lens_tensor)
        output_pad_sequence, _ = pad_packed_sequence(input_packed_sequence, padding_value = 1e-8)


        #再把位置调回来
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = torch.tensor([back_map[i] for i in range(len(sent_lens))])
        output_pad_sequence_reorder = output_pad_sequence.index_select(index=reorder_ind, dim=0)
        sent_divided_lens_reorder = sent_divided_lens_tensor.index_select(index=reorder_ind, dim=0).tolist()

        high_level_output = super().forward(output_pad_sequence_reorder, sent_divided_lens_reorder, False)

        return high_level_output

class SentToWordLSTM(nn.Module):
    def __init__(self, emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, sampling_teaching_force,embedding):
        super().__init__()
        self._dec_lstm = MultiLayerLSTMCells(n_hidden, n_hidden, n_layer, dropout=dropout)
        self._sampling_teaching_force = sampling_teaching_force
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding is not None:
            assert self._embedding.weight.size() == embedding.size()
            self._embedding.weight.data.copy_(embedding)

        self._projection = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        self._teaching_force_ratio = 1
        self._step_num = 0
        
        self._init_dec_h = nn.Parameter(
            torch.Tensor(n_layer, n_hidden)
        )
        self._init_dec_c = nn.Parameter(
            torch.Tensor(n_layer, n_hidden)
        )
        init.uniform_(self._init_dec_h, -INIT, INIT)
        init.uniform_(self._init_dec_c, -INIT, INIT)

    def forward(self, input_hidden_states, target, init_h, init_c):
        self._teaching_force_ratio = pow(0.999995, self._step_num)
        self._step_num += 1
        
        max_len = target.size()[1]
        
        if (init_h is None and init_c is None):
            #还可以试着全是None
            init_h = self._init_dec_h.repeat(input_hidden_states.size()[0], 1)
            init_c = self._init_dec_c.repeat(input_hidden_states.size()[0], 1)

      
        init_states = (torch.unsqueeze(init_h, 0).contiguous(),
                       torch.unsqueeze(init_c, 0).contiguous())  #传闻中变成连续块的函数

        #    #, input_hidden_states  这边瞎糊的
        states = init_states
        dec_out = input_hidden_states
        logits = [] 
        for i in range(max_len):
            #如果利用scheduled sampling方法，随机选择用真实还是生成的tok作为输入。
            sampling = False
            if (self._sampling_teaching_force and i!=0):
                ratio = random.random()
                if (ratio > self._teaching_force_ratio):
                    sampling = True

            if (sampling):
                tok = torch.max(lp, dim=1, keepdim=True)[1] #tok[:,0]
            else:
                tok = target[:, i:i+1]
            
            lp, states, dec_out = self._step(tok, states, dec_out)
            logits.append(lp)
        logit = torch.stack(logits, dim=1)   
        return logit           


    def _for_test(self, tok, states,input_states):
        
        if states is None:
            init_h = self._init_dec_h.repeat(input_states.size()[0], 1)
            init_c = self._init_dec_c.repeat(input_states.size()[0], 1)

            states = (torch.unsqueeze(init_h, 0).contiguous(),
            torch.unsqueeze(init_c, 0).contiguous())  #传闻中变成连续块的函数

        
        return self._step(tok,states,input_states)

    def _step(self, tok, states, input_states):
        lstm_in = torch.cat([self._embedding(tok).squeeze(1), input_states], dim=1) #这是原来的写法
        states = self._dec_lstm(lstm_in, states)
        lstm_out = states[0][-1]
        dec_out = self._projection(lstm_out)

        logit = torch.mm(dec_out, self._embedding.weight.t())
        logit = torch.log(F.softmax(logit, dim=-1) + 1e-8)
        
        return logit, states, dec_out

    def topk_step(self, tok, states, beam_size):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam*batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._dec_lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]
        
        dec_out = self._projection(lstm_out)

        logit = torch.mm(dec_out.contiguous().view(batch*beam, -1), self._embedding.weight.t())
        logit = torch.log(F.softmax(logit, dim=-1) + 1e-8).view(beam, batch, -1)

        k_logit, k_tok = logit.topk(k=beam_size, dim=-1)
        return k_tok, k_logit, (states, dec_out)

