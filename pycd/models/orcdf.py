from abc import abstractmethod

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, accuracy_score

# Orcdf Enhanced Module
class ORCDF_Extractor(nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9, mode='all', ssl_temp=0.8, ssl_weight=1e-2, **kwargs):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.mode = mode
        self.ssl_temp = ssl_temp
        self.ssl_weight = ssl_weight

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.gcn_drop = True
        self.graph_dict = ...

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)

        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }

        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)

        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.apply(self.initialize_weights)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_all_emb(self):
        stu_emb, exer_emb, know_emb = (self.__student_emb.weight,
                                       self.__exercise_emb.weight,
                                       self.__knowledge_emb.weight)
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb

    def convolution(self, graph):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self._graph_drop(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb

    def _common_forward(self, right, wrong):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for layer in range(self.gcn_layers):
            right_emb = torch.sparse.mm(self._graph_drop(right), right_emb)
            wrong_emb = torch.sparse.mm(self._graph_drop(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb[:self.student_num], out_emb[self.student_num:self.student_num + self.exercise_num], out_emb[
                                                                                                           self.exercise_num + self.student_num:]

    # def __common_forward(self, right, wrong):
    #     right_emb, wrong_emb = self.convolution(right), self.convolution(wrong)
    #     out = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
    #     return out[:self.student_num], out[self.student_num:self.student_num + self.exercise_num], out[
    #                                                                                                self.exercise_num + self.student_num:]

    def _dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def _graph_drop(self, graph):
        g_dropped = self._dropout(graph, self.keep_prob)
        return g_dropped

    def extract(self, student_id, exercise_id, q_mask):
        if 'dis' not in self.mode:
            stu_forward, exer_forward, know_forward = self._common_forward(self.graph_dict['right'],
                                                                           self.graph_dict['wrong'])
            stu_forward_flip, exer_forward_flip, know_forward_flip = self._common_forward(self.graph_dict['right_flip'],
                                                                                          self.graph_dict['wrong_flip'])
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = out[:self.student_num], out[
                                                                              self.student_num:self.student_num + self.exercise_num], out[
                                                                                                                                      self.exercise_num + self.student_num:]

        # stu_forward_flip_v1, exer_forward_flip_v1, know_forward_flip_v1 = self.__common_forward(self.graph_dict['right_v1'],
        #                                                                                self.graph_dict['wrong_v1'])
        # stu_forward_flip_v2, exer_forward_flip_v2, know_forward_flip_v2 = self.__common_forward(self.graph_dict['right_v2'],
        #                                                                                self.graph_dict['wrong_v2'])
        #
        extra_loss = 0

        def InfoNCE(view1, view2, temperature: float = 1.0, b_cos: bool = False):
            """
            Args:
                view1: (torch.Tensor - N x D)
                view2: (torch.Tensor - N x D)
                temperature: float
                b_cos (bool)

            Return: Average InfoNCE Loss
            """
            if b_cos:
                view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

            pos_score = (view1 @ view2.T) / temperature
            score = torch.diag(F.log_softmax(pos_score, dim=1))
            return -score.mean()

        # extra_loss
        if 'cl' not in self.mode:
            extra_loss = self.ssl_weight * (InfoNCE(stu_forward, stu_forward_flip, temperature=self.ssl_temp)
                                            + InfoNCE(exer_forward, exer_forward_flip,
                                                      temperature=self.ssl_temp))
            #print(extra_loss)

        # student_ts = self.transfer_student_layer(
        #     F.embedding(student_id, stu_forward)) if 'tf' not in self.mode else F.embedding(student_id, stu_forward)
        # diff_ts = self.transfer_exercise_layer(
        #     F.embedding(exercise_id, exer_forward)) if 'tf' not in self.mode else F.embedding(exercise_id, exer_forward)
        # knowledge_ts = self.transfer_knowledge_layer(know_forward) if 'tf' not in self.mode else know_forward
        student_ts = F.embedding(student_id, stu_forward)
        diff_ts = F.embedding(exercise_id, exer_forward)
        knowledge_ts = know_forward

        disc_ts = self.__disc_emb(exercise_id)
        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)
        return student_ts, diff_ts, disc_ts, knowledge_ts, {'extra_loss': extra_loss,
                                                            'knowledge_impact': knowledge_impact_ts}

    def get_flip_graph(self):
        def get_flip_data(data):
            import numpy as np
            np_response_flip = data.copy()
            column = np_response_flip[:, 2]
            probability = np.random.choice([True, False], size=column.shape,
                                           p=[self.graph_dict['flip_ratio'], 1 - self.graph_dict['flip_ratio']])
            column[probability] = 1 - column[probability]
            np_response_flip[:, 2] = column
            return np_response_flip

        response_flip = get_flip_data(self.graph_dict['response'])
        se_graph_right_flip, se_graph_wrong_flip = [self._create_adj_se(response_flip, is_subgraph=True)[i] for i in
                                                    range(2)]
        ek_graph = self.graph_dict['Q_Matrix']
        self.graph_dict['right_flip'], self.graph_dict['wrong_flip'] = self._final_graph(se_graph_right_flip,
                                                                                         ek_graph), self._final_graph(
            se_graph_wrong_flip, ek_graph)

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        if 'dis' not in self.mode:
            stu_forward, exer_forward, know_forward = self._common_forward(self.graph_dict['right'],
                                                                           self.graph_dict['wrong'])
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = (out[:self.student_num],
                                                       out[self.student_num:self.student_num + self.exercise_num],
                                                       out[self.exercise_num + self.student_num:])

        student_ts = self.transfer_student_layer(stu_forward) if 'tf' not in self.mode else stu_forward
        diff_ts = self.transfer_exercise_layer(exer_forward) if 'tf' not in self.mode else exer_forward
        knowledge_ts = self.transfer_knowledge_layer(know_forward) if 'tf' not in self.mode else know_forward

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]

    @staticmethod
    def _get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def _sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def _create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num))

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self._get_csr(train_stu_right, train_exer_right, shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self._get_csr(train_stu_wrong, train_exer_wrong, shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self._get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def _final_graph(self, se, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num: se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self._sp_mat_to_sp_tensor(adj_matrix).to(self.device)

# w = max(w, 0)
class _NoneNegClipper(object):
    def __init__(self):
        super(_NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

# Interaction Function
class KANCD_IF(nn.Module):
    def __init__(self, knowledge_num: int, latent_dim: int, hidden_dims: list, dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.k_diff_full = nn.Linear(self.latent_dim, 1, dtype=dtype).to(self.device)
        self.stat_full = nn.Linear(self.latent_dim, 1, dtype=dtype).to(self.device)

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype),
                        'activation0': nn.Tanh()
                    }
                )
            else:
                layers.update(
                    {
                        'dropout{}'.format(idx): nn.Dropout(p=self.dropout),
                        'linear{}'.format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype),
                        'activation{}'.format(idx): nn.Tanh()
                    }
                )
        layers.update(
            {
                'dropout{}'.format(len(self.hidden_dims)): nn.Dropout(p=self.dropout),
                'linear{}'.format(len(self.hidden_dims)): nn.Linear(
                    self.hidden_dims[len(self.hidden_dims) - 1], 1, dtype=self.dtype
                ),
                'activation{}'.format(len(self.hidden_dims)): nn.Sigmoid()
            }
        )

        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs['knowledge_ts']
        q_mask = kwargs["q_mask"]

        batch, dim = student_ts.size()
        stu_emb = student_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        exer_emb = diff_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
                                            - torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)) * q_mask
        return self.mlp(input_x).view(-1)

        # batch, dim = student_ts.size()
        # # Embedding 乘法准备
        # stu_emb = student_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        # exer_emb = diff_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        # knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        #
        # # 展平送入 Linear
        # stat_input = (stu_emb * knowledge_emb).view(-1, dim)
        # kdiff_input = (exer_emb * knowledge_emb).view(-1, dim)
        # stat_scores = self.stat_full(stat_input).view(batch, self.knowledge_num)
        # kdiff_scores = self.k_diff_full(kdiff_input).view(batch, self.knowledge_num)
        # input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(stat_scores) - torch.sigmoid(kdiff_scores)) * q_mask
        #
        # return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        self.eval()
        blocks = torch.split(torch.arange(mastery.shape[0]).to(device=self.device), 5)
        mas = []
        for block in blocks:
            batch, dim = mastery[block].size()
            stu_emb = mastery[block].view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
            knowledge_emb = knowledge.repeat(batch, 1).view(batch, self.knowledge_num, -1)
            mas.append(torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1))
        return torch.vstack(mas)

    def monotonicity(self):
        none_neg_clipper = _NoneNegClipper()
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

class ORCDF(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num, latent_dim=32, device: str = "cpu",
                 gcn_layers: int = 3, keep_prob=0.9,dtype=torch.float64, hidden_dims: list = None,
                 mode='all', flip_ratio=0.1, ssl_temp=0.8,ssl_weight=1e-2, if_type='kancd',
                 **kwargs):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.device = device
        self.mode = mode
        self.flip_ratio = flip_ratio

        #initialize orcdf extractor
        self.extractor = ORCDF_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
            mode=mode,
            ssl_temp=ssl_temp,
            ssl_weight=ssl_weight
        )

        # initialize interaction function
        self.inter_func = KANCD_IF(
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            hidden_dims=hidden_dims,
            dropout=0.5
        )

    def forward(self, user_id, item_id, q_matrix):
        student_ts, diff_ts, disc_ts, knowledge_ts, aux_info = self.extractor.extract(
            user_id, item_id, q_matrix
        )

        logits = self.inter_func.compute(
            student_ts=student_ts,
            diff_ts=diff_ts,
            disc_ts=disc_ts,
            q_mask=q_matrix,
            knowledge_ts=knowledge_ts
        )

        return logits, aux_info.get("extra_loss", 0.0)

    #[Apply monotonicity constraint to inter_func（kancd）.mlp layers
    def apply_clipper(self):
        if hasattr(self.inter_func, 'monotonicity'):
            self.inter_func.monotonicity()
    
    def on_epoch_start(self):
        """在每个epoch开始时调用，更新flip graph"""
        if hasattr(self.extractor, 'get_flip_graph') and hasattr(self.extractor, 'graph_dict'):
            self.extractor.get_flip_graph()

    def loss(self, preds, labels):
        if isinstance(preds, tuple):
            logits, extra_loss = preds
        else:
            logits = preds
            extra_loss = 0.0

        ce_loss = nn.BCELoss()(logits, labels.double())
        return ce_loss + extra_loss


def extract_response_array(dataset):
    """
        从 CDMDataset 提取为 numpy array，shape=(N, 3)，每行是 [user_id, item_id, correct]
    """
    records = []
    for u, q, _, r in dataset:
        records.append([
            int(u.item()) if torch.is_tensor(u) else int(u),
            int(q.item()) if torch.is_tensor(q) else int(q),
            int(r.item()) if torch.is_tensor(r) and r.numel() == 1 else int(r[0].item())
            ])
    return np.array(records, dtype=np.float64)