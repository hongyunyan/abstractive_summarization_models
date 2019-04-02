import torch
from torch import nn
from torch.nn import init

from rnn import lstm_encoder
from rnn import MultiLayerLSTMCells
from attention import step_attention
from utils import sequence_mean, len_mask, change_shape, change_reshape, change_reshape_decoder

from torch.nn import functional as F
from utils import reorder_sequence, reorder_lstm_states

from utils import EOA
from seq2seq import Seq2SeqSumm


INIT = 1e-2

class HierarchicalSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, embedding, dropout=0.0):
        super().__init__()

        self._bidirectional = bidirectional
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._n_hidden = n_hidden

        self._Seq2SeqSumm = Seq2SeqSumm(vocab_size, n_hidden, n_hidden, bidirectional, n_layer, dropout)
        self._WordToSentLSTM = WordToSentLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, embedding)
        self._SentToWordLSTM = SentToWordLSTM(emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, embedding)
    
    def forward(self, article_sents, article_lens, sent_lens,  abstract_sents, abs_lens):  
        # sent_
        #传给第一个函数的article是一个n个句子×m个word， art_lens为每个句子的长度的n维矩阵，应该还要再有一个矩阵，记录每个article有几个句子，这样可以将输出的hidden state转成对应的格式
        #先传给词到句子的那层, 输入source文本以及source每条文本的长度，返回每一句末尾的hidden_states和c值组成的矩阵,并且返回hidden_states中，每个文本的长度
        words_hidden_states, words_contexts = self._WordToSentLSTM(article_sents, sent_lens)  
        #转格式！
        #不会转格式啊嗷嗷啊！！！！
        #坑仿佛填上了！！ 根据上面那个batch，256，改道成一个文章数×句子长×256的矩阵 article_hidden_states !!!

        if self._bidirectional:
            hidden_states = torch.cat(words_hidden_states.chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】

        hidden_states = torch.stack([self._dec_h(h) for h in hidden_states], dim=0) #从 [1,batch,512] 到 [batch,256]
        hidden_states = hidden_states[-1]

        pad = 1e-8  #用来填充没有句子的地方的hidden
        #先生成一个文章数×文章最多的句子数的矩阵
        
        article_hidden_states = change_shape(hidden_states, article_lens, pad)
    
        #然后句子的输入开始过最基本的seq2seq层
        sent_dec_out, sent_h_out, sent_c_out = self._Seq2SeqSumm(article_hidden_states, article_lens, abs_lens)

        #感觉这边要设置一下，如何让sent数目输出的是正确的？？？？加入loss？？？？

        #坑坑坑来来来转格式了，从文章数×target句子长×256 到所有句子数×256  sentence_hidden_states!!
        sent_output = change_reshape([sent_dec_out, sent_h_out, sent_c_out], abs_lens)
        sentence_output_states, sentence_hidden_states, sentence_context_states = sent_output[:]

        #获得句子的每个hidden以后，一生多 生成每个句子, 然后每个生成的具体句子跟原始的target做loss，返回loss
        logit = self._SentToWordLSTM(sentence_output_states, abstract_sents, sentence_hidden_states, sentence_context_states)
        return logit
    
    
    def batch_decode(self, article_sents, article_lens, sent_lens, start, eos, max_sent, max_words):
        """ greedy decode support batching"""
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

        init_states = (torch.unsqueeze(sentence_hidden_states, 0).contiguous(),
                       torch.unsqueeze(sentence_context_states, 0).contiguous())
        states = init_states, sentence_output_states
        batch_size = sentence_output_states.size()[0]
        tok = torch.LongTensor([start]*batch_size).to(article_sents.device)

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
        for i in range(len(decoder_len)):
            sents = outputs[sent_num: sent_num + decoder_len[i]]
            sent_num = sent_num + decoder_len[i]
            article = []
            for sent in sents:
                if (sent[0] == EOA):
                    break
                sent_ids = []
                for word in sent:
                    sent_ids.append(word.item())
                article.append(sent_ids)
            articles_output.append(article)
        return articles_output

    def batched_beam_search(self, article_sents, article_lens, sent_lens,
                            go, eos, unk, max_len, beam_size, diverse=1.0):

        words_hidden_states, words_contexts = self._WordToSentLSTM(article_sents, sent_lens)  
        
        #抄换转格式！到时候给我苟回去！
        if self._bidirectional:
            hidden_states = torch.cat(words_hidden_states.chunk(2, dim=0), dim=2)  #从[2,batch,256] 到 【1,batch,512】

        hidden_states = torch.stack([self._dec_h(h) for h in hidden_states], dim=0) #从 [1,batch,512] 到 [batch,256]
        hidden_states = hidden_states[-1]
        pad = torch.ones(self._n_hidden) * 1e-8  #用来填充没有句子的地方的hidden

        max_sent_num = max(article_lens)
        #先生成一个文章数×文章最多的句子数的矩阵
        
        article_hidden_states = None
        count = 0
        for i in range(len(article_lens)):
            for j in range(article_lens[i]):
                if (i==0 and j==0):
                    article_hidden_states = torch.unsqueeze(hidden_states[count], 0)
                else:
                    article_hidden_states = torch.cat((article_hidden_states, torch.unsqueeze(hidden_states[count], 0)), dim=0)
                count += 1
            for k in range(max_sent_num - article_lens[i]):
                article_hidden_states = torch.cat((article_hidden_states, torch.unsqueeze(pad, 0)), dim=0)
        
        article_hidden_states = article_hidden_states.reshape(len(article_lens), max_sent_num, hidden_states.size()[1])

        #seq2seq
        sent_dec_out, sent_h_out, sent_c_out = self._Seq2SeqSumm(article_hidden_states, article_lens, abs_lens)

        #快乐继续换格式
        sentence_output_states = None
        sentence_hidden_states = None
        sentence_context_states = None

        for i in range(len(abs_lens)):
            for j in range(abs_lens[i]):
                if (i==0 and j==0):
                    sentence_output_states = torch.unsqueeze(sent_dec_out[i][j], 0)
                    sentence_hidden_states = torch.unsqueeze(sent_h_out[i][j], 0)
                    sentence_context_states = torch.unsqueeze(sent_c_out[i][j], 0)
                else:
                    sentence_output_states = torch.cat((sentence_output_states, torch.unsqueeze(sent_dec_out[i][j], 0)), dim=0)
                    sentence_hidden_states = torch.cat((sentence_hidden_states, torch.unsqueeze(sent_h_out[i][j], 0)), dim=0)
                    sentence_context_states = torch.cat((sentence_context_states, torch.unsqueeze(sent_c_out[i][j], 0)), dim=0)   #苟完了

        init_states = (torch.unsqueeze(sentence_hidden_states, 0).contiguous(),
                       torch.unsqueeze(sentence_context_states, 0).contiguous())
        states = init_states, sentence_output_states

        batch_size = sentence_output_states.size()[0]
        tok = torch.LongTensor([go]*batch_size).to(article.device) 



        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings

        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                    ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs


class SentToWordLSTM(nn.Module):
    def __init__(self, emb_dim, n_hidden, n_layer, bidirectional, dropout, vocab_size, embedding):
        super().__init__()
        self._dec_lstm = MultiLayerLSTMCells(n_hidden, n_hidden, n_layer, dropout=dropout)

        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding is not None:
            assert self._embedding.weight.size() == embedding.size()
            self._embedding.weight.data.copy_(embedding)

        self._projection = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )

    def forward(self, input_hidden_states, target, init_h, init_c):
        max_len = target.size()[1]

        # hidden_states = torch.cat([torch.unsqueeze(init_h, 0), torch.unsqueeze(init_c, 0)], dim=0)

        init_states = (torch.unsqueeze(init_h, 0).contiguous(),
                       torch.unsqueeze(init_c, 0).contiguous())  #传闻中变成连续块的函数

        states = init_states, input_hidden_states  #这边瞎糊的
        logits = [] 
        for i in range(max_len):
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


class WordToSentLSTM(nn.Module):
    def __init__(self, emb_dim, n_hidden, n_layer,
            bidirectional, dropout, vocab_size, embedding):
        super().__init__()


        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding is not None:
            assert self._embedding.weight.size() == embedding.size()
            self._embedding.weight.data.copy_(embedding)

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

    def lstm(self, sequence, seq_lens, init_enc_states):
        #输出为batch，hidden_dim
        batch_size = sequence.size(0)

        #注入embedding matrix
        sequence = sequence.transpose(0, 1)
        emb_sequence = self._embedding(sequence)

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
            

    def forward(self, article_sents, sent_lens):
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

        lstm_out, final_states  = self.lstm(article_sents, sent_lens, init_states)

        return final_states


