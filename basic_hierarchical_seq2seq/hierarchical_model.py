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
import torch.multiprocessing as mp

INIT = 1e-2

class HierarchicalSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, sampling_teaching_force, self_attn, hi_encoder, embedding, dropout=0.0):
        super().__init__()

        self._bidirectional = bidirectional
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._n_hidden = n_hidden
        self._self_attn = self_attn
        self._hi_encoder = hi_encoder
        self._Seq2SeqSumm = Seq2SeqSumm(vocab_size, n_hidden, n_hidden, bidirectional, n_layer, dropout, embedding)
        self._WordToSentLSTM = WordToSentLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, self_attn, embedding)
        self._SentToWordLSTM = SentToWordLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, sampling_teaching_force, embedding)
        self._HierarchicalWordToSentLSTM = HierarchicalWordToSentLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, self_attn, embedding)
    
    def forward(self, article_sents, article_lens, sent_lens,  abstract_sents, abs_lens):  
        start_timestamp = time()
        #传给第一个函数的article是一个n个句子×m个word， art_lens为每个句子的长度的n维矩阵，应该还要再有一个矩阵，记录每个article有几个句子，这样可以将输出的hidden state转成对应的格式
        #先传给词到句子的那层, 输入source文本以及source每条文本的长度，返回每一句末尾的hidden_states和c值组成的矩阵,并且返回hidden_states中，每个文本的长度

        #尝试一个多层的wordtosent，天哪我在干什么orz

        wordToSentModel = self._HierarchicalWordToSentLSTM if self._hi_encoder else self._WordToSentLSTM

        if (self._self_attn):
            sent_output = wordToSentModel(article_sents, sent_lens)  
            wordtosent_timestamp = time()

            hidden_states = torch.stack([self._dec_h(h) for h in sent_output], dim=0) #从 [batch,512] 到 [batch,256]
        else:
            words_hidden_states = wordToSentModel(article_sents, sent_lens)  
            wordtosent_timestamp = time()

            #转格式！
            #不会转格式啊嗷嗷啊！！！！
            #坑仿佛填上了！！ 根据上面那个batch，256，改道成一个文章数×句子长×256的矩阵 article_hidden_states !!!

            if self._bidirectional:
                hidden_states = torch.cat(words_hidden_states.chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】

            hidden_states = torch.stack([self._dec_h(h) for h in hidden_states], dim=0)[-1] #从 [1,batch,512] 到 [batch,256]

        #转格式！
        #不会转格式啊嗷嗷啊！！！！
        #坑仿佛填上了！！ 根据上面那个batch，256，改道成一个文章数×句子长×256的矩阵 article_hidden_states !!!

        pad = 1e-8  #用来填充没有句子的地方的hidden
        #先生成一个文章数×文章最多的句子数的矩阵
        
        article_hidden_states = change_shape(hidden_states, article_lens, pad)
        
        change_shape_timestamp = time()
        #然后句子的输入开始过最基本的seq2seq层
        sent_dec_out, sent_h_out, sent_c_out = self._Seq2SeqSumm(article_hidden_states, article_lens, abs_lens)
        seq2seq_timestamp = time()
        #感觉这边要设置一下，如何让sent数目输出的是正确的？？？？加入loss？？？？
        #坑坑坑来来来转格式了，从文章数×target句子长×256 到所有句子数×256  sentence_hidden_states!!
        sent_output = change_reshape([sent_dec_out, sent_h_out, sent_c_out], abs_lens)
        sentence_output_states, sentence_hidden_states, sentence_context_states = sent_output[:]
        change_reshape_timestamp = time()
        #获得句子的每个hidden以后，一生多 生成每个句子, 然后每个生成的具体句子跟原始的target做loss，返回loss
        logit = self._SentToWordLSTM(sentence_output_states, abstract_sents, sentence_hidden_states, sentence_context_states)

        senttoword_timestamp = time()

        # print("word_to_sent_timestamp\n", wordtosent_timestamp - start_timestamp)
        # print("change_shape_timestamp\n", change_shape_timestamp - wordtosent_timestamp)
        # print("seq2seq_timestamp\n", seq2seq_timestamp - change_shape_timestamp)
        # print("change_reshape_timestamp\n", change_reshape_timestamp - seq2seq_timestamp)
        # print("sent_to_word_timestamp\n", senttoword_timestamp - change_reshape_timestamp)

        return logit
    
    
    def batch_decode(self, article_sents, article_lens, sent_lens, start, max_sent, max_words):
        """ greedy decode support batching"""
        
        if (self._self_attn):
             sent_output = self._WordToSentLSTM(article_sents, sent_lens)  
             hidden_states = torch.stack([self._dec_h(h) for h in sent_output], dim=0) #从 [batch,512] 到 [batch,256]
        else:
            words_hidden_states= self._WordToSentLSTM(article_sents, sent_lens)  
            if self._bidirectional:
                hidden_states = torch.cat(words_hidden_states.chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】
            hidden_states = torch.stack([self._dec_h(h) for h in hidden_states], dim=0) #从 [1,batch,512] 到 [batch,256]
            hidden_states = hidden_states[-1]
    
        pad = 1e-8  #用来填充没有句子的地方的hidden

        article_hidden_states = change_shape(hidden_states, article_lens, pad)

        #这边要改成encoder和decoder分离，先让已知的数据穿过seq2seq的encoder然后获得参数开始decoder 句子vec，然后再用句子vec decoder出sent vec
        #或者这边应该调用seq2seq的beam_decoder函数，让他自己decoder好返回给我下一步decoder的参数
        sent_dec_out, sent_h_out, sent_c_out = self._Seq2SeqSumm.batch_decode(article_hidden_states, article_lens, max_sent)

        #快乐继续换格式
        sent_output = change_reshape_decoder([sent_dec_out, sent_h_out, sent_c_out])
        sentence_output_states, sentence_hidden_states, sentence_context_states = sent_output[:]

        init_states = (torch.unsqueeze(sentence_hidden_states, 0).contiguous(),
                       torch.unsqueeze(sentence_context_states, 0).contiguous())
        states = init_states, sentence_output_states

        tok = torch.cat([torch.arange(max_sent)] * len(article_lens), dim=0).to(article_sents.device)

        outputs = None
        for i in range(max_words):
            logit, states = self._SentToWordLSTM._step(tok, states)
            tok = torch.max(logit, dim=1, keepdim=True)[1]  #挣扎一下维度对不对
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

    def batched_beam_search(self, article_sents, article_lens, sent_lens,
                            start, max_sent, max_words, eos, beam_size, diverse=1.0):

        words_hidden_states, words_contexts = self._WordToSentLSTM(article_sents, sent_lens)  

        #抄换转格式！到时候给我苟回去！
        if self._bidirectional:
            hidden_states = torch.cat(words_hidden_states.chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】

        hidden_states = torch.stack([self._dec_h(h) for h in hidden_states], dim=0) #从 [1,batch,512] 到 [batch,256]
        hidden_states = hidden_states[-1]

        pad = 1e-8  #用来填充没有句子的地方的hidden

        article_hidden_states = change_shape(hidden_states, article_lens, pad)

        #这边要改成encoder和decoder分离，先让已知的数据穿过seq2seq的encoder然后获得参数开始decoder 句子vec，然后再用句子vec decoder出sent vec
        #或者这边应该调用seq2seq的beam_decoder函数，让他自己decoder好返回给我下一步decoder的参数
        sent_dec_out, sent_h_out, sent_c_out, decoder_len = self._Seq2SeqSumm.batch_decode(article_hidden_states, article_lens, max_sent)

        #快乐继续换格式
        sent_output = change_reshape_decoder([sent_dec_out, sent_h_out, sent_c_out], decoder_len)
        sentence_output_states, sentence_hidden_states, sentence_context_states = sent_output[:]

        batch_size = sentence_output_states.size()[0]

        h = sentence_hidden_states.unsqueeze(0)
        c = sentence_context_states.unsqueeze(0)
        prev = sentence_output_states

        all_beams = [bs.init_beam(start.to(article_sents.device), (h[:, i, :], c[:, i, :], prev[i])) for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_words):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                       torch.stack([prev for _, prev in all_states], dim=1))
            # token.masked_fill_(token >= vsize, unk)
            
            topk, k_logit, states = self._SentToWordLSTM.topk_step(token, states, beam_size)


            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos, EOA, 
                    topk[:, batch_i, :], k_logit[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :])
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs, finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]

        #这个神奇的output里面有finish的beam size的结果

        articles_output = []
        sent_num = 0
        for i in range(len(decoder_len)):
            sents = outputs[sent_num: sent_num + decoder_len[i]]
            sent_num = sent_num + decoder_len[i]
            article = []
            for sent in sents:
                if (sent[0].sequence[0] == EOA):
                    break
                sent_ids = []
                for word in sent[0].sequence:
                    sent_ids.append(word)
                article.append(sent_ids)
            articles_output.append(article)


        return articles_output


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

    def forward(self, input_hidden_states, target, init_h, init_c):
        self._teaching_force_ratio = pow(0.999995, self._step_num)
        self._step_num += 1
        
        max_len = target.size()[1]

        init_states = (torch.unsqueeze(init_h, 0).contiguous(),
                       torch.unsqueeze(init_c, 0).contiguous())  #传闻中变成连续块的函数

        states = init_states, input_hidden_states  #这边瞎糊的
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
            
            lp, states = self._step(tok, states)
            logits.append(lp)
        logit = torch.stack(logits, dim=1)   
        return logit           

    def _step(self, tok, states):
        
        prev_states, prev_dec_out = states
        lstm_in = torch.cat([self._embedding(tok).squeeze(1), prev_dec_out], dim=1)
        states = self._dec_lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        dec_out = self._projection(lstm_out)

        logit = torch.mm(dec_out, self._embedding.weight.t())
        logit = torch.log(F.softmax(logit, dim=-1) + 1e-8)
        
        return logit, (states, dec_out)

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

        self._lstm_layer = nn.LSTM(input_size = emb_dim, hidden_size= n_hidden, num_layers = n_layer, bidirectional = bidirectional, dropout = dropout)

        if bidirectional == True:
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * n_hidden  ,2 * n_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * n_hidden))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2* n_hidden, 1))
        else:
            #先不支持单向
            pass
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
            return sent_output
        else:
            return final_states[0]


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


       

