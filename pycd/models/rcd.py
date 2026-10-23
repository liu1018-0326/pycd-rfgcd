# PYCD/pycd/models/rcd.py
# -*- coding: utf-8 -*-
# Combined RCD Model Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- GraphLayer --------

class GraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


# -------- Fusion --------

class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # graph structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)

        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, args.knowledge_n, args.knowledge_n)
        self.undirected_gat = GraphLayer(self.undirected_g, args.knowledge_n, args.knowledge_n)
        self.k_from_e = GraphLayer(self.k_from_e, args.knowledge_n, args.knowledge_n)
        self.e_from_k = GraphLayer(self.e_from_k, args.knowledge_n, args.knowledge_n)
        self.u_from_e = GraphLayer(self.u_from_e, args.knowledge_n, args.knowledge_n)
        self.e_from_u = GraphLayer(self.e_from_u, args.knowledge_n, args.knowledge_n)

        self.k_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        k_directed = self.directed_gat(kn_emb)
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)

        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.u_from_e(e_u_graph)
        e_from_u_graph = self.e_from_u(e_u_graph)

        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        D = k_from_e_graph[self.exer_n:]
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)
        score1 = self.k_attn_fc1(concat_c_1)
        score2 = self.k_attn_fc2(concat_c_2)
        score3 = self.k_attn_fc3(concat_c_3)
        score = F.softmax(torch.cat([torch.cat([score1, score2], dim=1), score3], dim=1), dim=1)
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C + score[:, 2].unsqueeze(1) * D

        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0: self.exer_n]
        C = e_from_u_graph[0: self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_n:]

        return kn_emb, exer_emb, all_stu_emb


# -------- RCD Model --------

class RCD(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)

        super(RCD, self).__init__()

        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        self.FusionLayer1 = Fusion(args, local_map)
        self.FusionLayer2 = Fusion(args, local_map)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        self.prednet_full3 = nn.Linear(args.knowledge_n, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        batch_stu_emb = all_stu_emb2[stu_id]
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        batch_exer_emb = exer_emb2[exer_id]
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0], kn_emb2.shape[1])

        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output.squeeze(-1)
        # return output
    
    def get_all_knowledge_emb(self):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        batch_stu_emb = all_stu_emb2
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        batch_exer_emb = exer_emb2
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0], kn_emb2.shape[1])

        return torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
    
    def loss(self, preds, labels):
        """
        preds: Tensor of shape [B, 1] — 表示正类概率
        labels: Tensor of shape [B] or [B, 1] — 标签（0或1）
        origin implement: NLLLoss + log_softmax
        """
        if preds.dim() > 1 and preds.size(1) == 1:
            preds = preds.squeeze(1)
        if labels.dim() > 1:
            labels = labels.squeeze(1)

        loss_fn = nn.BCELoss()
        return loss_fn(preds, labels.float())  # 注意要转换成 float 类型



class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

