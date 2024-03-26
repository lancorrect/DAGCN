
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, kl_loss,  pooled_output= self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        return logits, kl_loss


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_reshape, src_mask, aspect_mask = inputs
        h1, h2, kl_loss, pooled_output = self.gcn(inputs)
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)
        outputs1 = (h1 * aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2 * aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2, kl_loss, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(self.opt, opt.attention_heads, self.bert_dim)
        self.wa = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.wa.append(nn.Linear(input_dim, self.mem_dim))

        self.ws = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.ws.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_reshape, src_mask, aspect_mask = inputs
        src_mask = src_mask.unsqueeze(-2) 
        batch = src_mask.size(0)
        len = src_mask.size()[2]
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids).values()
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.bert_dim)
        aspect_outs = gcn_inputs*aspect_mask

        aspect_scores, s_attn = self.attn(gcn_inputs, gcn_inputs, src_mask, aspect_outs, aspect_mask)
        aspect_score_list = [attn_adj.squeeze(1) for attn_adj in torch.split(aspect_scores, 1, dim=1)]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(s_attn, 1, dim=1)]
        adj_ag = None

        aspect_score_avg = None
        adj_s = None

        # Average Aspect-aware Attention scores
        for i in range(self.attention_heads):
            if aspect_score_avg is None:
                aspect_score_avg = aspect_score_list[i]
            else:
                aspect_score_avg += aspect_score_list[i]
        aspect_score_avg = aspect_score_avg / self.attention_heads

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
        adj_s = src_mask.transpose(1, 2) * adj_s

        # distance based weighted matrix
        adj_reshape = torch.exp((-1.0) * self.opt.alpha * adj_reshape)

        # aspect-aware attention * distance based weighted matrix
        distance_mask = (
                    aspect_score_avg > torch.ones_like(aspect_score_avg) * self.opt.beta)
        adj_reshape = adj_reshape.masked_fill(distance_mask, 1).cuda()
        adj_ag = (adj_reshape * aspect_score_avg).type(torch.float32)

        # KL divergence
        kl_loss = F.kl_div(adj_ag.softmax(-1).log(), adj_s.softmax(-1), reduction='sum')
        kl_loss = torch.exp((-1.0) * kl_loss * self.opt.gama)

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

        return outputs_ag, outputs_s, kl_loss, pooled_output


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