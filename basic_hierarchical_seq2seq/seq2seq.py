import torch
from torch import nn
from torch.nn import init

from rnn import lstm_encoder
from rnn import MultiLayerLSTMCells
from attention import step_attention
from utils import sequence_mean, len_mask, special_word_num

from torch.nn import functional as F

INIT = 1e-2


class Seq2SeqSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout=0.0, embedding=None):
        super().__init__()

        self._embedding = nn.Embedding(vocab_size, int(emb_dim/2), padding_idx=0)
        if embedding is not None:
            assert self._embedding.weight.size() == embedding.size()
            self._embedding.weight.data.copy_(embedding)

        self._enc_lstm = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )
        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        output_dim = int(n_hidden/2)

        # vanillat lstm / LNlstm
        self._dec_lstm = MultiLayerLSTMCells(n_hidden, n_hidden, n_layer, dropout=dropout)

        # project encoder final states to decoder initial states
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        # multiplicative attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        # project decoder output to emb_dim, then
        # apply weight matrix from embedding layer
        
        self._projection = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, output_dim, bias=False)
        )
        # functional object for easier usage
        self._decoder = AttentionalLSTMDecoder(self._dec_lstm, self._attn_wq, self._projection, self._embedding)
    
    def forward(self, article, art_lens, tar_lens):

        attention, init_dec_states = self.encode(article, art_lens)  #(34,81,256) ((1,34,256),(1,34,256),（34,128))
        #init_dec_states传的是[(h,c).attn_out],后者不知道在干吗
        #attention感觉是encoder的结果乘以attention的矩阵
        mask = len_mask(art_lens, attention.device).unsqueeze(-2) #改个格式 [34,1,81]
        dec_out_all = self._decoder((attention, mask), init_dec_states, tar_lens)
        return dec_out_all

    def encode(self, article, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )  #(2,34,256)
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )#扩展states,but为啥是这个states,以及正常情况下也是并行处理的吗？随机初始化

        enc_art, final_states = lstm_encoder(article, self._enc_lstm, art_lens, init_enc_states)

        if self._enc_lstm.bidirectional:
            h, c = final_states

            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),  #从[2,34,256] 到 【1,34,512】
                torch.cat(c.chunk(2, dim=0), dim=2)
            )

        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0) # 把[1,34,512]过一个linear 变成【1,34,256】，大概就是不想用原生的lstm输出，要经过改变?
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)  #(1,34,256)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1) #根据attention weight算出c  (81,34,512) * (512, 256) -- (81,34,256) --transpose (34,81,256)  我感觉应该是为了回归正常的size

        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1  #(34,256*2) -- 经过一个网络 --- （34,128)
        )) #感觉有种[h,c]的感觉？？没懂这步在干嘛

        return attention, (init_dec_states, init_attn_out)

    def batch_decode(self, article, art_lens, max_len):
        #返回dec_out,h,c还有每个article的lens
        batch_size = len(art_lens)
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask)
        
        h_output = []
        c_output = []
        dec_output = []
        states = init_dec_states
        for i in range(max_len):
            tok = torch.tensor(special_word_num + i).expand(states[1].size()[0], 1).cuda()
            states = self._decoder.decode_step(tok, states, attention)
            (h,c), dec_out = states

            h_output.append(h[0])
            c_output.append(c[0])
            dec_output.append(dec_out)

        #     #怎么判断lens呢先空着呗
        # lens = [max_len] * batch_size
        #     #填充好，然后最后判断lens
        return dec_output, h_output, c_output

    def decode(self, article, go, eos, max_len):
        attention, init_dec_states = self.encode(article)
        attention = (attention, None)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
        return outputs, attns

class AttentionalLSTMDecoder(object):
    def __init__(self, lstm, attn_w, projection, embedding):
        super().__init__()
        self._lstm = lstm
        self._attn_w = attn_w
        self._projection = projection
        self._embedding = embedding

    def __call__(self, attention, init_states, tar_lens):
        states = init_states
 
        dec_out_all = []
        h_all = []
        c_all = []

        #给sentence level的decoder输入标记这是第几句话的vec
        for i in range(max(tar_lens)):  #感觉是在模拟time stamp
            tok = torch.tensor(special_word_num + i).expand(states[1].size()[0], 1).cuda()
            states= self._step(tok, states, attention)
            (h,c), dec_out = states

            dec_out_all.append(dec_out)
            h_all.append(h[0])
            c_all.append(c[0])

        #返回sentence level的所有dec_out
        dec_out_all = torch.stack(dec_out_all, dim=0).transpose(0,1)
        h_all = torch.stack(h_all, dim=0).transpose(0,1)
        c_all = torch.stack(c_all, dim=0).transpose(0,1)
        
        #返回sentence level的所有dec_out
        return dec_out_all, h_all, c_all #这样就是batch×length×n_hidden了

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in =  torch.cat([self._embedding(tok).squeeze(1), prev_out],dim=1)
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))

        states = (states, dec_out)

        return states

    def decode_step(self, tok, states, attention):
        states= self._step(tok, states, attention)
        
        return states
