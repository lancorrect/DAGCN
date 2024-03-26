import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy

class GCNClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        in_dim = opt.hidden_dim * 2
        self.opt = opt
        self.gcn_model = GCNAbsaModel(opt, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, kl_loss = self.gcn_model(inputs)
        final_output = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_output)

        adj_ag = self.adj_ag
        adj_s = self.adj_s
        adj_s_T = adj_s.transpose(1,2)
        identity = torch.eye(adj_s.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_s.size(0), adj_s.size(1), adj_s.size(1))
        ortho = adj_ag@adj_s_T

        penal1 = (torch.norm(ortho - identity) / adj_s.size(0)).cuda()
        penal2 = (adj_s.size(0) / torch.norm(adj_s - adj_ag)).cuda()
        penal = 0.25 * penal1 + 0.25 * penal2

        return logits, penal
    
    @property
    def aspect_attention_scores(self):
        return self.gcn_model.aspect_attention_scores
    
    @property
    def adj_ag(self):
        return self.gcn_model.adj_ag

    @property
    def adj_s(self):
        return self.gcn_model.adj_s

class GCNAbsaModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None        # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None    # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj_reshape = inputs           # unpack inputs
        maxlen = max(l.data)
        mask = mask[:, :maxlen]

        h1, h2, kl_loss = self.gcn(inputs)
        
        # avg pooling asp feature, h:(16,28,50)
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                        # aspect words num, mask:(16,85), asp_wn:(16,1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)    # mask for h, mask:(16,85,50)
        outputs1 = (h1 * mask).sum(dim=1) / asp_wn                        # mask h1
        outputs2 = (h2 * mask).sum(dim=1) / asp_wn                        # mask h2

        return outputs1, outputs2, kl_loss
    
    @property
    def aspect_attention_scores(self):
        return self.gcn.aspect_attention_scores

    @property
    def adj_ag(self):
        return self.gcn.adj_ag

    @property
    def adj_s(self):
        return self.gcn.adj_s

class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.embed_dim+opt.pos_dim+opt.post_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True,
                           dropout=opt.rnn_dropout, bidirectional=opt.bidirect)

        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden

        # drop out
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.wa = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.wa.append(nn.Linear(input_dim, self.mem_dim))

        self.ws = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.in_dim if j == 0 else self.mem_dim
            self.ws.append(nn.Linear(input_dim, self.mem_dim))

        # aspect-aware attention
        self.attention_heads = opt.attention_heads
        self.aspect_attention_scores = None
        self.attn = MultiHeadAttention(self.opt, self.attention_heads, self.mem_dim * 2)

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size, emb_class='pos'):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj_reshape = inputs           # unpack inputs
        maxlen = max(l.data)
        src_mask = (tok != 0).unsqueeze(-2)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs = embs + [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs = embs + [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        self.rnn.flatten_parameters()

        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l.to('cpu'), tok.size()[0], 'pos'))

        # aspect-fusion attention
        aspect_ids = aspect_indices(mask)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim * 2)
        mask = mask[:, :maxlen, :]
        aspect_outs = gcn_inputs * mask

        aspect_scores, s_attn = self.attn(gcn_inputs, gcn_inputs, src_mask, aspect_outs, mask)
        aspect_score_list = [attn_adj.squeeze(1) for attn_adj in torch.split(aspect_scores, 1, dim=1)]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(s_attn, 1, dim=1)]

        aspect_score_avg = None
        adj_s = None

        # Average Aspect-aware Attention scores
        for i in range(self.attention_heads):
            if aspect_score_avg is None:
                aspect_score_avg = aspect_score_list[i]
            else:
                aspect_score_avg += aspect_score_list[i]
        aspect_score_avg = aspect_score_avg / self.attention_heads
        self.aspect_attention_scores = aspect_score_avg

        # * Average Multi-head Attention matrices
        for i in range(self.attention_heads):
            if adj_s is None:
                adj_s = attn_adj_list[i]
            else:
                adj_s += attn_adj_list[i]
        adj_s = adj_s / self.attention_heads

        for j in range(adj_s.size(0)):
            adj_s[j] -= torch.diag(torch.diag(adj_s[j]))
            adj_s[j] += torch.eye(adj_s[j].size(0)).cuda()  # self-loop
        adj_s = mask_ * adj_s
        self.adj_s = adj_s

        # distance based weighted matrix
        adj_reshape = adj_reshape[:, :maxlen, :maxlen]
        adj_reshape = torch.exp(self.opt.alpha*(-1.0)*adj_reshape)

        # aspect-aware attention * distance based weighted matrix
        distance_mask = (aspect_score_avg > torch.ones_like(aspect_score_avg)*self.opt.beta)
        adj_reshape = adj_reshape.masked_fill(distance_mask, 1).cuda()
        adj_ag = (adj_reshape * aspect_score_avg).type(torch.float32)
        self.adj_ag = adj_ag

        # KL divergence
        kl_loss = F.kl_div(adj_ag.softmax(-1).log(), adj_s.softmax(-1), reduction='sum')
        kl_loss = torch.exp((-1.0)*kl_loss*self.opt.gama)
        
        # gcn layer
        denom_s = adj_s.sum(2).unsqueeze(2) + 1
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_s = gcn_inputs
        outputs_ag = gcn_inputs

        for l in range(self.layers):
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.wa[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            Ax_s = adj_s.bmm(outputs_s)
            AxW_s = self.ws[l](Ax_s)
            AxW_s = AxW_s / denom_s
            gAxW_s = F.relu(AxW_s)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine1), torch.transpose(gAxW_s, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_s, self.affine2), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            gAxW_ag, gAxW_s = torch.bmm(A1, gAxW_s), torch.bmm(A2, gAxW_ag)
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag
            outputs_s = self.gcn_drop(gAxW_s) if l < self.layers - 1 else gAxW_s

        return outputs_ag, outputs_s, kl_loss

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def aspect_indices(mask):
    aspect_id_mask = copy.deepcopy(mask).cpu()
    aspect_id_mask = torch.nonzero(aspect_id_mask).numpy()
    aspect_id_dict = {}
    for elem in aspect_id_mask:
        key = elem[0]
        value = elem[1]
        if key in aspect_id_dict.keys():
            aspect_id_dict[key].append(value)
        else:
            aspect_id_dict[key] = [value]
    
    aspect_ids = list(aspect_id_dict.values())
    return aspect_ids


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, opt, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.opt = opt
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)
    
    def attention(self, query, key, mask, dropout):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        s_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            s_attn = dropout(s_attn)

        return s_attn
    
    def aspect_attention(self, key, aspect, aspect_mask):

        if self.opt.fusion is True:
            aspect = self.query(aspect)
            new_aspect_shape = aspect.shape[:2] + (self.h, self.d_k,)
            aspect = aspect.view(new_aspect_shape)
            aspect = aspect.permute(0, 2, 1, 3)

            aspect_raw_scores = torch.matmul(aspect, key.transpose(-2, -1))
            aspect_mask = aspect_mask[:,:,0].unsqueeze(1).unsqueeze(-1).repeat(1, self.h, 1, 1)
            aspect_raw_scores = (aspect_raw_scores + self.bias) * aspect_mask
            aspect_scores = torch.sigmoid(aspect_raw_scores)
        else:
            aspect_scores = torch.tanh(
                torch.add(torch.matmul(torch.matmul(aspect, self.weight_m), key.transpose(-2, -1)), self.bias))
        
        return aspect_scores
    
    def forward(self, query, key, mask, aspect, aspect_mask):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        aspect_scores = None
        aspect_scores = self.aspect_attention(key, aspect, aspect_mask)
        self_attn = self.attention(query, key, mask, self.dropout)

        return aspect_scores, self_attn

