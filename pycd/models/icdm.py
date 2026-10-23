import wandb as wb
import numpy as np
import scipy.sparse as sp
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from tqdm import tqdm
import warnings
from dgl.base import DGLError
import dgl.function as fn
from dgl.utils import check_eq_shape, expand_as_pair
from pycd.train.trainer import Trainer, EarlyStopping
import os
import networkx as nx
from torch.utils.data import TensorDataset, DataLoader 
from dgl.nn import GraphConv

print("ICDM模型111")

def transform(q: torch.tensor, user, item, score, batch_size, dtype=torch.float32):
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        q[item, :],
        torch.tensor(score, dtype=dtype)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def get_r_matrix(np_test, stu_num, prob_num, new_idx=None):
    if new_idx is None:
        r = -1 * np.ones(shape=(stu_num, prob_num))
        for i in range(np_test.shape[0]):
            s = int(np_test[i, 0])
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    else:
        r = -1 * np.ones(shape=(stu_num, prob_num))

        for i in range(np_test.shape[0]):
            s = new_idx.index(int(np_test[i, 0]))
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    return r


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def l2_loss(*weights):
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2)) / w.shape[0]

    return 0.5 * loss


def dgl2tensor(g):
    nx_graph = g.to_networkx()
    adj_matrix = nx.adjacency_matrix(nx_graph).todense()
    tensor = torch.from_numpy(adj_matrix)
    return tensor


def concept_distill(matrix, concept):
    coeff = 1.0 / torch.sum(matrix, dim=1)
    concept = matrix.to(concept.dtype) @ concept
    concept_distill = concept * coeff[:, None]
    return concept_distill


def get_subgraph(g, id, device):
    return dgl.in_subgraph(g, id).to(device)


warnings.filterwarnings('ignore')


def compute_laplacian_pe(graph, k=16):
    """
    Compute normalized Laplacian eigenvectors as topology-aware positional encoding.
    Return shape: [num_nodes, k].
    """
    n = int(graph.num_nodes())
    if k <= 0:
        return torch.zeros((n, 0), dtype=torch.float32)
    if n == 0:
        return torch.zeros((0, k), dtype=torch.float32)
    if n == 1:
        return torch.zeros((1, k), dtype=torch.float32)

    k_eff = min(k, n - 1)
    try:
        adj = graph.adjacency_matrix(scipy_fmt='csr').astype(np.float64)
        deg = np.asarray(adj.sum(1)).flatten()
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(deg, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        lap = sp.eye(n, format='csr') - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        _, eigenvectors = sp.linalg.eigsh(lap, k=k_eff + 1, which='SM', tol=1e-10)
        pe = eigenvectors[:, 1:k_eff + 1]
    except Exception:
        pe = np.random.randn(n, k_eff)

    if k_eff < k:
        pad = np.zeros((n, k - k_eff))
        pe = np.concatenate([pe, pad], axis=1)
    return torch.from_numpy(pe).float()


def build_graph4CE(config: dict):
    q = config['q']
    q = q.detach().cpu().numpy()
    know_num = int(config['know_num'])
    exer_num = int(config['prob_num'])
    node = exer_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    indices = np.where(q != 0)
    for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
        edge_list.append((int(know_id + exer_num), int(exer_id)))
        edge_list.append((int(exer_id), int(know_id + exer_num)))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


def build_graph4SE(config: dict, mode='tl'):
    if mode == 'tl':
        data = config['np_train']
    elif mode == 'ind_train':
        data = config['np_train_old']
    else:
        data = np.vstack((config['np_train_old'], config['np_train_new']))

    stu_num = config['stu_num']
    exer_num = config['prob_num']
    node = stu_num + exer_num
    g_right = dgl.DGLGraph()
    g_right.add_nodes(node)
    g_wrong = dgl.DGLGraph()
    g_wrong.add_nodes(node)

    right_edge_list = []
    wrong_edge_list = []

    for index in range(data.shape[0]):
        stu_id = int(data[index, 0])
        exer_id = int(data[index, 1])
        if int(float(data[index, 2])) == 1:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                right_edge_list.append((int(stu_id), int(exer_id + stu_num)))
                right_edge_list.append((int(exer_id + stu_num), int(stu_id)))
            else:
                right_edge_list.append((int(exer_id + stu_num), int(stu_id)))
        else:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                wrong_edge_list.append((int(stu_id), int(exer_id + stu_num)))
                wrong_edge_list.append((int(exer_id + stu_num), int(stu_id)))
            else:
                wrong_edge_list.append((int(exer_id + stu_num), int(stu_id)))
    right_src, right_dst = tuple(zip(*right_edge_list))
    wrong_src, wrong_dst = tuple(zip(*wrong_edge_list))
    g_right.add_edges(right_src, right_dst)
    g_wrong.add_edges(wrong_src, wrong_dst)
    return g_right, g_wrong


def build_graph4SC(config: dict, mode='tl'):
    if mode == 'tl':
        data = config['np_train']
    elif mode == 'ind_train':
        data = config['np_train_old']
    else:
        data = np.vstack((config['np_train_old'], config['np_train_new']))
    stu_num = int(config['stu_num'])
    know_num = int(config['know_num'])
    q = config['q']
    q = q.detach().cpu().numpy()
    node = stu_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    sc_matrix = np.zeros(shape=(stu_num, know_num))
    for index in range(data.shape[0]):
        stu_id = data[index, 0]
        exer_id = data[index, 1]
        concepts = np.where(q[int(exer_id)] != 0)[0]
        for concept_id in concepts:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                if sc_matrix[int(stu_id), int(concept_id)] != 1:
                    edge_list.append((int(stu_id), int(concept_id + stu_num)))
                    edge_list.append((int(concept_id + stu_num), int(stu_id)))
                    sc_matrix[int(stu_id), int(concept_id)] = 1
            else:
                if mode != 'involve':
                    if sc_matrix[int(stu_id), int(concept_id)] != 1:
                        edge_list.append((int(concept_id + stu_num), int(stu_id)))
                        sc_matrix[int(stu_id), int(concept_id)] = 1
                else:
                    if sc_matrix[int(stu_id), int(concept_id)] != 1:
                        edge_list.append((int(stu_id), int(concept_id + stu_num)))
                        sc_matrix[int(stu_id), int(concept_id)] = 1
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g



class SAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=True,
            norm=None,
            activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                                             : graph.num_dst_nodes()
                                             ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                        degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = h_neigh
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = graph.dstdata["neigh"]
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class SAGENet(nn.Module):
    def __init__(self, dim, layers_num=2, type='mean', device='cpu', drop=True):
        super(SAGENet, self).__init__()
        self.drop = drop
        self.type = type
        self.layers = []
        for i in range(layers_num):
            if type == 'mean':
                self.layers.append(SAGEConv(in_feats=dim, out_feats=dim, aggregator_type=type).to(device))

    def forward(self, g, h):
        outs = [h]
        tmp = h
        from dgl import DropEdge
        for index, layer in enumerate(self.layers):
            drop = DropEdge(p=0.05 + 0.1 * index)
            if self.drop:
                if self.training:
                    g = drop(g)
                if self.type != 'mean':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            else:
                if self.type != 'mean':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            outs.append(tmp / (1 + index))
        res = torch.sum(torch.stack(
            outs, dim=1), dim=1)
        return res


class DenoisingGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, device, k_svd=32):
        super(DenoisingGCN, self).__init__()
        self.gcn_softmax = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.k_svd = k_svd
        self.device = device

    def softmax_edge_gcn(self, g: dgl.DGLGraph, node_emb: torch.Tensor):
        g = g.to(self.device)
        node_emb = node_emb.to(self.device)
        with torch.no_grad():
            edge_src, edge_dst = g.edges()
            edge_sim = (node_emb[edge_src] * node_emb[edge_dst]).sum(dim=1)
            edge_prob = F.softmax(edge_sim, dim=0)
            g.edata["w"] = edge_prob

        def weighted_msg(edges):
            return {"m": edges.src["h"] * edges.data["w"].unsqueeze(-1)}

        with g.local_scope():
            g.ndata["h"] = node_emb
            g.update_all(weighted_msg, fn.sum("m", "h_agg"))
            return self.gcn_softmax(g, node_emb)

    def forward(self, g: dgl.DGLGraph, node_emb: torch.Tensor):
        out = self.softmax_edge_gcn(g, node_emb)
        return out


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, f1, f2, f3):
        feats = torch.stack([f1, f2, f3], dim=1)
        Q = self.query(feats)  
        K = self.key(feats)
        V = self.value(feats)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / (self.dim ** 0.5), dim=-1)
        fused = torch.matmul(attn, V) 
        return fused.mean(dim=1)


class TopoAttentionFusion(nn.Module):
    def __init__(self, dim, pe_dim):
        super().__init__()
        self.dim = dim
        self.pe_proj = nn.Linear(pe_dim, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, f1, f2, f3, pe):
        feats = torch.stack([f1, f2, f3], dim=1)
        pe_emb = self.pe_proj(pe).unsqueeze(1)
        feats = feats + pe_emb
        q = self.query(feats)
        k = self.key(feats)
        v = self.value(feats)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.dim ** 0.5), dim=-1)
        fused = torch.matmul(attn, v)
        output = self.layer_norm(fused.mean(dim=1) + f1)
        return self.out_proj(output)
    

class IGNet(nn.Module):
    def __init__(self, stu_num, prob_num, know_num, dim, graph, norm_adj=None, inter_layers=3, hidden=512,
                 device='cuda',
                 gcnlayers=3, agg_type='mean', cdm_type='glif', khop=2, dropout=0.2, pe_dim=16,
                 stu_pe=None, exer_pe=None):
        super().__init__()
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.dim = dim
        self.graph = graph
        self.device = device
        self.norm_adj = norm_adj.to(self.device)
        self.cdm_type = cdm_type
        self.khop = khop
        self.dropout = dropout
        self.pe_dim = pe_dim

        self.stu_emb = nn.Embedding(self.stu_num, self.dim)
        self.exer_emb_right = nn.Embedding(self.prob_num, self.dim)
        self.exer_emb_wrong = nn.Embedding(self.prob_num, self.dim)
        self.know_emb = nn.Embedding(self.know_num, self.dim)

        self.disc_emb = nn.Embedding(self.prob_num, 1)
        self.s_e = nn.Embedding(self.prob_num+self.stu_num, self.dim)


        self.global_user = nn.Parameter(torch.zeros(size=(1, self.dim)))
        self.gcn_layers = gcnlayers
        self.E_C_right = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)
        self.E_C_wrong = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)
        self.S_C = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)

        self.Involve_Matrix = dgl2tensor(self.graph['I'])[:self.stu_num, self.stu_num:].to(self.device)
        self.transfer_stu_layer = nn.Linear(self.dim, self.know_num)
        self.transfer_exer_layer = nn.Linear(self.dim, self.know_num)
        self.transfer_concept_layer = nn.Linear(self.dim, self.know_num)

        self.change_latent_stu_mirt = nn.Linear(self.know_num, 16)
        self.change_latent_exer_mirt = nn.Linear(self.know_num, 16)
        self.change_latent_stu_irt = nn.Linear(self.know_num, 1)
        self.change_latent_exer_irt = nn.Linear(self.know_num, 1)

        self.D_model = DenoisingGCN(in_feats=self.dim, hidden_feats=self.dim, device='cuda')
        self.gcn_reconstruct = GraphConv(in_feats=self.dim, out_feats=self.dim, allow_zero_in_degree=True)
        self.q_proj = nn.Linear(know_num, dim)
        self.k_proj = nn.Linear(know_num, dim)
        self.v_proj = nn.Linear(know_num, dim)
        self.out_proj = nn.Linear(dim, know_num)
        self.dropout = nn.Dropout(dropout)
        self.c_w = nn.Linear(self.dim, self.dim)
        self.t_w = nn.Linear(self.dim, self.dim)
        self.attn_c = AttentionFusion(self.dim)
        self.topo_fusion_stu = TopoAttentionFusion(self.dim, self.pe_dim)
        self.topo_fusion_exer = TopoAttentionFusion(self.dim, self.pe_dim)
        self.stu_emb_a = nn.Embedding(self.stu_num, self.know_num)

        if stu_pe is None:
            stu_pe = torch.zeros((self.stu_num, self.pe_dim), dtype=torch.float32)
        if exer_pe is None:
            exer_pe = torch.zeros((self.prob_num, self.pe_dim), dtype=torch.float32)
        self.register_buffer("stu_pe", stu_pe.float(), persistent=False)
        self.register_buffer("exer_pe", exer_pe.float(), persistent=False)
        
        layers = []
        for i in range(inter_layers):
            layers.append(nn.Linear(self.know_num if i == 0 else hidden // pow(2, i - 1), hidden // pow(2, i)))
            layers.append(nn.Dropout(p=0.3))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden // pow(2, inter_layers - 1), 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        BatchNorm_names = ['layers.{}.weight'.format(4 * i + 1) for i in range(3)]
        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name and param.dim() >= 2:
                if name not in BatchNorm_names:
                    nn.init.xavier_normal_(param)

    def compute(self, emb):
        all_emb = emb
        embs = [emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb.to(torch.float32))
            embs.append(all_emb)  
        out_embs = torch.mean(torch.stack(embs, dim=1), dim=1)

        return out_embs



    def forward(self, stu_id, exer_id, knowledge_point):
        concept_id = torch.where(knowledge_point != 0)[1].to(self.device)
        concept_id_S = concept_id + torch.full(concept_id.shape, self.stu_num).to(self.device)
        concept_id_E = concept_id + torch.full(concept_id.shape, self.prob_num).to(self.device)
        exer_id_S = exer_id + torch.full(exer_id.shape, self.stu_num).to(self.device)

        subgraph_node_id_Q = torch.unique(torch.cat((exer_id.detach().cpu(), concept_id_E.detach().cpu()), dim=-1))
        subgraph_node_id_R = torch.unique(torch.cat((stu_id.detach().cpu(), exer_id_S.detach().cpu()), dim=-1))
        subgraph_node_id_I = torch.unique(torch.cat((stu_id.detach().cpu(), concept_id_S.detach().cpu()), dim=-1))

        R_subgraph_Right = get_subgraph(self.graph['right'], subgraph_node_id_R, device=self.device)
        R_subgraph_Wrong = get_subgraph(self.graph['wrong'], subgraph_node_id_R, device=self.device)
        I_subgraph = get_subgraph(self.graph['I'], subgraph_node_id_I, device=self.device)
        Q_subgraph = get_subgraph(self.graph['Q'], subgraph_node_id_Q, device=self.device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(self.device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(self.device)


        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        stu_info = self.stu_emb.weight

        S_C = torch.cat([stu_info, concept_info]).to(self.device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(self.device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(self.device)


        S_E_info_right = self.D_model(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.gcn_reconstruct(R_subgraph_Wrong, S_E_wrong)
        S_C_info = self.S_C(I_subgraph, S_C)

        w = torch.sigmoid(self.c_w(E_C_info_right[:self.prob_num]) + self.t_w(E_C_info_wrong[:self.prob_num]))
        E_blend = w * E_C_info_right[:self.prob_num] + (1 - w) * E_C_info_wrong[:self.prob_num]
        E_forward = self.topo_fusion_exer(
            E_C_info_right[:self.prob_num],
            E_C_info_wrong[:self.prob_num],
            E_blend,
            self.exer_pe
        )
        C_forward = self.attn_c(E_C_info_right[self.prob_num:], E_C_info_wrong[self.prob_num:], S_C_info[self.stu_num:])
        S_forward = self.topo_fusion_stu(
            S_E_info_right[:self.stu_num],
            S_E_info_wrong[:self.stu_num],
            S_C_info[:self.stu_num],
            self.stu_pe
        )
        emb = torch.cat([S_forward, E_forward]).to(self.device)
        disc = torch.sigmoid(self.disc_emb(exer_id))


        out = self.compute(emb)
        out = out.to(torch.float32)
        S_forward, E_forward, C_forward = self.transfer_stu_layer(out[:self.stu_num]), self.transfer_exer_layer(out[self.stu_num:]), self.transfer_concept_layer(C_forward)
        exer_concept_distill = concept_distill(knowledge_point, C_forward)

        state_s = disc * (torch.sigmoid(S_forward[stu_id] * exer_concept_distill) - torch.sigmoid(E_forward[exer_id] * exer_concept_distill)) * knowledge_point
        


        return self.layers(state_s.to(torch.float32)).view(-1)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)


class ICDM(CDM):
    def __init__(self, stu_num, prob_num, know_num, dim=64, device='cuda:0', graph=None, gcn_layers=3, agg_type='mean',
                 weight_reg=0.05, wandb=True, cdm_type='glif', khop=2, exist_idx=None, new_idx=None, dropout=0.2,
                 pe_dim=16):
        super(ICDM, self).__init__()
        self.net = None
        self.know_num = know_num
        self.prob_num = prob_num
        self.stu_num = stu_num
        self.device = device
        self.dim = dim
        self.wandb = wandb
        self.agg_type = agg_type
        self.graph = graph
        self.gcn_layers = gcn_layers
        self.weight_reg = weight_reg
        self.cdm_type = cdm_type
        self.khop = khop
        self.mas_list = []
        self.exist_idx = exist_idx
        self.new_idx = new_idx
        self.dropout = dropout
        self.pe_dim = pe_dim
        self.train_times = 0

    def train(self, np_train, np_val, np_test, np_test_new=None, q=None, batch_size=None, exp_dir=None, wandb_instance=None, epoch=10, lr=0.0005):

        train_data, val_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)  
            for _ in [np_train, np_val, np_test]
        ]


        def log_metrics(metrics):
            if wandb_instance:
                 wandb_instance.log(metrics)


        def get_adj_matrix(tmp_adj):
            adj_mat = tmp_adj + tmp_adj.T
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            return adj_matrix

        def sp_mat_to_sp_tensor(sp_mat):
            coo = sp_mat.tocoo().astype(np.float64)
            indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
            return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float32).coalesce()

        def create_adj_mat():
            n_nodes = self.stu_num + self.prob_num
            train_stu = np_train[:, 0]
            train_exer = np_train[:, 1]
            ratings = np.ones_like(train_stu, dtype=np.float64)
            tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.stu_num)), shape=(n_nodes, n_nodes))
            return sp_mat_to_sp_tensor(get_adj_matrix(tmp_adj))

        def build_pe_graph():
            right_g = self.graph['right'].to('cpu')
            wrong_g = self.graph['wrong'].to('cpu')
            r_src, r_dst = right_g.edges()
            w_src, w_dst = wrong_g.edges()
            src = torch.cat([r_src, w_src], dim=0)
            dst = torch.cat([r_dst, w_dst], dim=0)
            pe_g = dgl.graph((src, dst), num_nodes=(self.stu_num + self.prob_num))
            pe_g = dgl.to_simple(pe_g)
            return pe_g

        norm_adj = create_adj_mat()
        pe_graph = build_pe_graph()
        full_pe = compute_laplacian_pe(pe_graph, k=self.pe_dim)
        stu_pe = full_pe[:self.stu_num, :]
        exer_pe = full_pe[self.stu_num:, :]
        self.net = IGNet(stu_num=self.stu_num, prob_num=self.prob_num, know_num=self.know_num, dim=self.dim,
                        device=self.device, graph=self.graph, norm_adj=norm_adj, gcnlayers=self.gcn_layers,
                        agg_type=self.agg_type, cdm_type=self.cdm_type, khop=self.khop, dropout=self.dropout,
                        pe_dim=self.pe_dim, stu_pe=stu_pe, exer_pe=exer_pe).to(self.device)
        self.net = self.net.to(torch.float32)

        r = get_r_matrix(np_val, self.stu_num, self.prob_num)
        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        early_stop = EarlyStopping(patience=5, mode='max')

        exp_dir = exp_dir 
        ckpt_path = os.path.join(exp_dir, 'model.pth')
        print(f"model save to: {ckpt_path}")
        best_metric = None
        best_epoch = None


        for epoch_i in range(1, epoch+1):
            losses = []
            bce_losses = []
            reg_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Train"):
                batch_count += 1
                stu_id, exer_id, knowledge_emb, y = batch_data  
                stu_id: torch.Tensor = stu_id.to(self.device)
                exer_id: torch.Tensor = exer_id.to(self.device)
                knowledge_emb = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                tmp_E_right = self.net.exer_emb_right.weight
                tmp_E_wrong = self.net.exer_emb_wrong.weight
                reg_loss = l2_loss(       
                    tmp_E_right[exer_id],
                    tmp_E_wrong[exer_id]
                )
                pred = self.net.forward(stu_id, exer_id, knowledge_emb)
                bce_loss = bce_loss_function(pred, y)
                bce_losses.append(bce_loss.mean().item())
                total_loss = bce_loss + self.weight_reg * reg_loss  
                reg_losses.append((self.weight_reg * reg_loss).detach().cpu().numpy())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.net.apply_clipper()
                losses.append(total_loss.mean().item())


            loss = sum(losses) / len(losses)
            if wandb_instance:
                log_metrics({
                    'epoch': epoch_i
                })

            auc, accuracy, rmse, f1, doa = self.eval(val_data, q=q, r=r, val_test="Eval")
            print(f"[Epoch {epoch_i}] Train Loss: {loss:.4f}, Val Metric(Auc): {auc}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, Val RMSE: {rmse:.4f}, Val DOA: {doa:.4f}")

            if ckpt_path is not None:
                if best_metric is None or auc > best_metric:
                    best_metric = auc
                    best_epoch = epoch_i
                    torch.save(self.net.state_dict(), ckpt_path)
                    print(f"  → Saved best model to {ckpt_path}")

            if early_stop is not None:
                if early_stop.step(auc):
                    print(f"  → Early stopping at epoch {epoch_i}")
                    break

        if ckpt_path is not None:
            self.net.load_state_dict(torch.load(ckpt_path))
            r = get_r_matrix(np_test, self.stu_num, self.prob_num)
            auc, accuracy, rmse, f1, doa = self.eval(test_data, q=q, r=r, val_test="Test")

            
            return auc, accuracy, rmse, f1, doa, ckpt_path, best_metric, best_epoch

    def eval(self, val_data, q=None, r=None, r_new=None, val_test=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        y_true_new, y_pred_new = [], []
        for batch_data in tqdm(val_data, f"{val_test}"):
            user_id, item_id, know_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            know_emb = know_emb.to(self.device)
            pred: torch.Tensor = self.net.forward(user_id, item_id, knowledge_point=know_emb)
            if self.new_idx is not None:
                for index, user in enumerate(user_id.detach().cpu().tolist()):
                    if user in self.new_idx:
                        y_true_new.append(y.tolist()[index])
                        y_pred_new.append(pred.detach().cpu().tolist()[index])
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
                np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,), 0
