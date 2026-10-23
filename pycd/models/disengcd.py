import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import scipy as sp
from torch import Tensor
from typing import List, Dict, Tuple,Union
# NAS搜索meta multigraph部分
class Op(nn.Module):
    '''
    operation for one link in the DAG search space
    '''

    def __init__(self, k):
        super(Op, self).__init__()
        self.k = k   # nc: k=1   lr: k=2


    def forward(self, x, adjs, ws):
        num_op = len(ws)
        num = int(num_op//self.k)
        idx = random.sample(range(num_op),num)
       
        return sum(ws[i] * torch.spmm(expand_sparse_to_target_dims_torch(adjs[i],(x.shape[0],x.shape[0])), x) for i in range(num_op) if i in idx) / num #self.k  #num   #self.k


class Cell(nn.Module):
    '''
    the DAG search space
    '''
    def __init__(self, n_step, n_hid_prev, n_hid, cstr, k, use_norm = True, use_nl = True, ratio = 1):
        super(Cell, self).__init__()
        
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step               #* number of intermediate states (i.e., K)
        self.norm = nn.LayerNorm(n_hid, elementwise_affine = False) if use_norm is True else lambda x : x
        self.use_nl = use_nl
        assert(isinstance(cstr, list))
        self.cstr = cstr                   #* type constraint
        self.ratio = ratio
        op = Op(k)

        self.ops_seq = nn.ModuleList()     #* state (i - 1) -> state i, 1 <= i < K,  AI,  seq: sequential
        for i in range(1, self.n_step):
            self.ops_seq.append(op)
        self.ops_res = nn.ModuleList()     #* state j -> state i, 0 <= j < i - 1, 2 <= i < K,  AIO,  res: residual
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(op)

        self.last_seq = op               #* state (K - 1) -> state K,  /hat{A}
        self.last_res = nn.ModuleList()    #* state i -> state K, 0 <= i < K - 1,  /hat{A}IO
        for i in range(self.n_step - 1):
            self.last_res.append(op)


    

    def forward(self, x, adjs, ws_seq, ws_res):
        #assert(isinstance(ws_seq, list))
        #assert(len(ws_seq) == 2)

        x = self.affine(x)
        states = [x]
        offset = 0
        edge = 1
        for i in range(self.n_step - 1):
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i])   #! exclude zero Op
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append((seqi + self.ratio * resi)/edge)
        #assert(offset == len(self.ops_res))

        adjs_cstr = [adjs[i] for i in self.cstr]
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1])

        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i]) for i, h in enumerate(states[:-1]))
        output = self.norm((out_seq + self.ratio * out_res)/edge)
        if self.use_nl:
            output = F.gelu(output)
        return output


class Model_paths(nn.Module):

    def __init__(self,device, in_dim, n_hid, num_node_types, n_adjs, n_classes, n_steps, ratio, cstr, k, lambda_seq, lambda_res, attn_dim = 64, use_norm = True, out_nl = True):
        super(Model_paths, self).__init__()
        self.device = device
        self.num_node_types = num_node_types
        self.cstr = cstr  
        self.n_adjs = n_adjs  
        self.n_hid = n_hid   
        self.ws = nn.ModuleList()          #* node type-specific transformation
        self.lambda_seq = lambda_seq
        self.lambda_res = lambda_res
        for i in range(num_node_types): 
            self.ws.append(nn.Linear(in_dim, n_hid))  
        assert(isinstance(n_steps, list))  #* [optional] combine more than one meta data?
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):  
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, cstr, k, use_norm = use_norm, use_nl = out_nl, ratio = ratio))  # self.metas contions 1 Cell

        self.as_seq = []                   #* arch parameters for ops_seq    k<K and i=k-1   AI
        self.as_last_seq = []              #* arch parameters for last_seq   k=K and i=k-1  /hat{A}
        for i in range(len(n_steps)):
            if n_steps[i] > 1:  # not for
                ai = 1e-3 * torch.randn(n_steps[i] - 1, (n_adjs - 1))   #! exclude zero Op   torch.randn(3, 5)  AI
                ai = ai.to(self.device)
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            ai_last = 1e-3 * torch.randn(len(cstr))  # torch.randn(2)  edge related to the evaluation  /hat{A}   actually /hat{A} I
            ai_last = ai_last.to(self.device)
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)

        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        self.as_res = []                  #* arch parameters for ops_res    k<K and i<k-1    AIO
        self.as_last_res = []             #* arch parameters for last_res   k=K and i<k-1    /hat{A}IO
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)  # (3,6)  AIO
                ai = ai.to(self.device)
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)
            
            if n_steps[i] > 1:
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1) 
                ai_last = ai_last.to(self.device)
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)

        assert(ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (1 + n_steps[0]) * n_steps[0] // 2)


        #* [optional] combine more than one meta data?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim) 
        self.attn_fc2 = nn.Linear(attn_dim, 1)  

        self.classifier = nn.Linear(n_hid, n_classes)

    def forward(self, node_feats, node_types, adjs):
        hid = torch.zeros((node_types.shape[0], self.n_hid)).to(self.device)
        for i in range(self.num_node_types):
            idx = (node_types == i)
            hid[idx] = self.ws[i](node_feats[idx])
        temps = []
        attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))  # softmax here
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1)) 
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, ws_res)  # cell
            temps.append(hidi)  
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1]))) # attni.shape   
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)  
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1) 
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)  # attns.unsqueeze(dim=-1) * hids 
        logits = self.classifier(out) 
        return logits


    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)

        return alphas

    def getid(self, seq_res, lam):
        seq_softmax = None if seq_res is None else F.softmax(seq_res, dim=-1)

        length = seq_res.size(-1)
        if len(seq_res.shape) == 1:
            max = torch.max(seq_softmax, dim=0).values
            min = torch.min(seq_softmax, dim=0).values
            threshold = lam * max + (1 - lam) * min
            return [k for k in range(length) if seq_softmax[k].item()>=threshold]
        max = torch.max(seq_softmax, dim=1).values
        min = torch.min(seq_softmax, dim=1).values
        threshold = lam * max + (1 - lam) * min
        res = [[k for k in range(length) if seq_softmax[j][k].item() >= threshold[j]] for j in range(len(seq_softmax))]
        return res

    def sample_final(self, eps):
        '''
        to sample one candidate edge type per link
        '''
        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)): 
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1), size=self.as_seq[i].size()[:-1]).to(self.device))
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)).to(self.device))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1), size=self.as_res[i].size()[:-1]).to(self.device))  # self.as_res[0]: shape [3,6]   high:  6   size :3
                temp.append(None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1), size=self.as_last_res[i].size()[:-1]).to(self.device)) # self.as_last_res[0]: shape [3,3]   high:  3   size :3
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                seq = self.getid(self.as_seq[i], self.lambda_seq)
                last_seq = self.getid(self.as_last_seq[i], self.lambda_seq)
                temp.append(seq)
                temp.append(last_seq)
                idxes_seq.append(temp)

            for i in range(len(self.metas)):
                temp = []
                res = self.getid(self.as_res[i], self.lambda_res)
                last_res = self.getid(self.as_last_res[i], self.lambda_res)
                temp.append(res)
                temp.append(last_res)
                idxes_res.append(temp)
        return idxes_seq, idxes_res

    
    def parse(self):
        '''
        to derive a meta data indicated by arch parameters
        '''
        idxes_seq, idxes_res = self.sample_final(0.)

        msg_seq = []; msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [[self.cstr[item] for item in idxes_seq[i][1]]]
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0] + map_seq) #idxes_seq[0][0]+idxes_seq[0][1]

            assert(len(msg_seq[i]) == self.metas[i].n_step)
            temp_res = []
            if idxes_res[i][1] is not None:
                for res in idxes_res[i][1]:
                    temp = []
                    for item in res:
                        if item < len(self.cstr):
                            temp.append(self.cstr[item])
                        else:
                            assert(item == len(self.cstr))
                            temp.append(self.n_adjs - 1)
                    temp_res.append(temp)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0] + temp_res   # idxes_res[0][0]+idxes_res[0][1]
            assert(len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2)
            msg_res.append(temp_res)
        

        return msg_seq, msg_res
    

#main model
class DisenGCD(nn.Module):
    def __init__(self,args):
        all_map = args.all_map
        node_types = args.node_types
        local_map=args.local_map

        self.device = args.device
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.stu_n = args.user_n
        self.predDisenGCD_input_len = self.knowledge_dim
        self.predDisenGCD_len1, self.predDisenGCD_len2 = 512, 256
        self.map = all_map
        self.node_type=args.node_types
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)

        super(DisenGCD,self).__init__()

        #DisenGCDwork structure
        self.stu_emb = nn.Embedding(self.stu_n,self.knowledge_dim).to(self.device)
        self.kn_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)  
        self.exer_emb = nn.Embedding(self.exer_n, self.knowledge_dim)  

        self.index = torch.LongTensor(list(range(self.stu_n))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)  
        self.k_index = torch.LongTensor(list(range( self.knowledge_dim))).to(self.device)  


        #学生的嵌入
        gpu_id = 0 if args.device == 'cpu' else int(args.device.split(':')[-1]) if ':' in str(args.device) else 0
        self.FusionLayer1 = Model_paths(gpu_id,args.knowledge_n,args.n_hid,3,len(self.map),args.knowledge_n, [4],
                                  args.ratio,[3,4],args.k,args.lam_seq,args.lam_res)                         
        self.FusionLayer3 = Fusion(args, local_map)
        self.FusionLayer4 = Fusion(args, local_map)

        self.predDisenGCD_full3 = PosLinear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(name)
                nn.init.xavier_normal_(param)


    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.stu_emb(self.index).to(self.device)
        exer_emb = self.exer_emb(self.exer_index).to(self.device)
        kn_emb = self.kn_emb(self.k_index).to(self.device)

  
        all_emb = torch.cat((all_stu_emb,exer_emb,kn_emb),0)
        all_stu_emb1 = self.FusionLayer1(all_emb, self.node_type, self.map)
        all_stu_emb2 = all_stu_emb1[self.knowledge_dim+self.exer_n:self.knowledge_dim+self.exer_n+self.stu_n, :]
        #all_stu_emb2 = all_stu_emb1[1747:3714, :]

        exer_emb1,kn_emb1 = self.FusionLayer3(exer_emb,kn_emb)
        exer_emb2,kn_emb2 = self.FusionLayer4(exer_emb,kn_emb1)



        batch_exer_emb = exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])

        # get batch student data
        batch_stu_emb = all_stu_emb2[stu_id]  # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb.shape[0],
                                                                     kn_emb.shape[1])


        # C认知诊
        alpha = batch_exer_vector + kn_vector
        betta = batch_stu_vector + kn_vector
        o = torch.sigmoid(self.predDisenGCD_full3(alpha * betta))


        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)             
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)              
        output = sum_out / count_of_concept
        return output
    def loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        计算损失。
        """
        return nn.NLLLoss()(pred, label.long())
    def get_all_knowledge_emb(self):
          
        all_stu_emb = self.stu_emb(self.index).to(self.device)
        exer_emb = self.exer_emb(self.exer_index).to(self.device)
        kn_emb = self.kn_emb(self.k_index).to(self.device)

  
        all_emb = torch.cat((all_stu_emb,exer_emb,kn_emb),0)
        all_stu_emb1 = self.FusionLayer1(all_emb, self.node_type, self.map)
        all_stu_emb2 = all_stu_emb1[self.knowledge_dim+self.exer_n:self.stu_n + self.exer_n+self.stu_n, :]
        #all_stu_emb2 = all_stu_emb1[1747:3714, :]

        exer_emb1,kn_emb1 = self.FusionLayer3(exer_emb,kn_emb)
        exer_emb2,kn_emb2 = self.FusionLayer4(exer_emb,kn_emb1)



        batch_exer_emb = exer_emb2  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])

        # get batch student data
        batch_stu_emb = all_stu_emb2  # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb.shape[0],
                                                                     kn_emb.shape[1])


        # C认知诊
        
        return batch_stu_vector + kn_vector





class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = args.device
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.user_n
        self.stu_dim = self.knowledge_dim

        # data structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)

        self.e_from_k = local_map['e_from_k'].to(self.device)


        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, args.knowledge_n, args.knowledge_n)
        self.undirected_gat = GraphLayer(self.undirected_g, args.knowledge_n, args.knowledge_n)
        self.e_from_k = GraphLayer(self.e_from_k, args.knowledge_n, args.knowledge_n)  # src: k

        self.k_from_e = GraphLayer(self.k_from_e, args.knowledge_n, args.knowledge_n)  # src: e

        self.k_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

    def forward(self, exer_emb,kn_emb):
        k_directed = self.directed_gat(kn_emb)  
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        e_from_k_graph = self.e_from_k(e_k_graph)
        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        score1 = self.k_attn_fc1(concat_c_1) 
        score2 = self.k_attn_fc2(concat_c_2)  
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)
                         
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C


        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0:self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        exer_emb = exer_emb + score1[:, 0].unsqueeze(1) * B

        return exer_emb,kn_emb
    
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
    

class PosLinear(nn.Linear):
    def forward(self, input):
        weight = 2 * F.relu(-self.weight) + self.weight
        return F.linear(input, weight, self.bias)

def normalize_sym(adj):
    """用于对称归一化邻接矩阵"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx):
    """用于行归一化稀疏矩阵，它接收一个Scipy稀疏矩阵sparse_mx作为输入，并将其转换为PyTorch稀疏张量表示。"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return sp.sparse.coo_matrix(mx).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """用于将Scipy稀疏矩阵转换为PyTorch稀疏张量"""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def crop_tensors_to_smallest_square(tensor1: np.ndarray, tensor2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将两个 NumPy 张量裁剪到它们共同的最小平方形状 (n x n)。
    裁剪从张量的左上角开始。

    参数:
    tensor1 (np.ndarray): 第一个 NumPy 张量。
    tensor2 (np.ndarray): 第二个 NumPy 张量。

    返回:
    tuple[np.ndarray, np.ndarray]: 一个包含两个裁剪后张量的元组。
                                    两个张量的形状都将是 (n, n, ...)，
                                    其中 '...' 代表任何原始的额外维度。

    异常:
    ValueError: 如果任一张量的维度少于2。
    """
    if tensor1.ndim < 2 or tensor2.ndim < 2:
        raise ValueError("两个张量都必须至少有2个维度 (例如：高度和宽度)。")

    # 获取每个张量的前两个维度 (通常是高度和宽度)
    h1, w1 = tensor1.shape[0], tensor1.shape[1]
    h2, w2 = tensor2.shape[0], tensor2.shape[1]

    # 确定最小的边长 n
    # n 必须小于或等于所有四个维度值 (h1, w1, h2, w2)
    n = min(h1, w1, h2, w2)

    # 从左上角开始裁剪两个张量到 n x n
    # NumPy 的切片[:n, :n] 会自动处理剩余的维度
    cropped_tensor1 = tensor1[:n, :n]
    cropped_tensor2 = tensor2[:n, :n]

    return cropped_tensor1, cropped_tensor2


def expand_sparse_to_target_dims_torch(
    tensor_sparse: torch.Tensor,
    target_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    将一个 n1*n2 的 PyTorch COO 稀疏张量扩展到 m1*m2 大小。
    新增的行和列在稀疏表示中是隐式的（即没有对应的非零元素）。
    原始非零元素的位置和值保持不变。

    参数:
    tensor_sparse (torch.Tensor): 输入的 n1*n2 PyTorch COO 稀疏张量。
                                  必须是 torch.sparse_coo_tensor 类型。
    target_shape (Tuple[int, int]): 目标形状 (m1, m2)。
                                    m1 必须 >= n1 (原始高度), m2 必须 >= n2 (原始宽度)。

    返回:
    torch.Tensor: 扩展后的 m1*m2 COO 稀疏张量，与输入张量在同一设备上，
                  且具有相同的值数据类型。

    异常:
    TypeError: 如果 tensor_sparse 不是 PyTorch 张量、不是稀疏张量、
               不是 COO 格式，或者 target_shape 不是包含两个整数的元组。
    ValueError: 如果输入张量不是二维的，
                或者 target_shape 的任一维度为负数，
                或者 target_shape 的维度小于原始稀疏张量的对应维度。
    RuntimeError: 如果 tensor_sparse 未聚合且无法访问其索引（尽管函数会尝试聚合）。
    """
    # --- Input Validation ---
    if not isinstance(tensor_sparse, torch.Tensor):
        raise TypeError("输入 tensor_sparse 必须是 PyTorch 张量。")
    if not tensor_sparse.is_sparse:
        raise TypeError("输入张量 tensor_sparse 必须是稀疏的。此函数针对 COO 格式。")
    if tensor_sparse.layout != torch.sparse_coo:
        raise TypeError(f"输入稀疏张量 tensor_sparse 必须是 COO 格式 (torch.sparse_coo), 得到 {tensor_sparse.layout}。")
    
    if not (isinstance(target_shape, tuple) and len(target_shape) == 2 and
            isinstance(target_shape[0], int) and isinstance(target_shape[1], int)):
        raise TypeError("target_shape 必须是一个包含两个整数的元组 (m1, m2)。")

    if tensor_sparse.ndim != 2:
        raise ValueError(f"输入稀疏张量 tensor_sparse 必须是二维的，但得到了 {tensor_sparse.ndim} 维。")

    n1_orig, n2_orig = tensor_sparse.shape
    m1_target, m2_target = target_shape

    if m1_target < 0 or m2_target < 0:
        raise ValueError(f"目标形状的维度不能是负数，得到 {target_shape}。")
    
    if m1_target < n1_orig:
        raise ValueError(f"目标高度 m1 ({m1_target}) 不能小于原始稀疏张量的高度 n1 ({n1_orig})。")
    if m2_target < n2_orig:
        raise ValueError(f"目标宽度 m2 ({m2_target}) 不能小于原始稀疏张量的宽度 n2 ({n2_orig})。")

    # --- Handle Identical Shape ---
    if n1_orig == m1_target and n2_orig == m2_target:
        return tensor_sparse.clone() # Return a clone for consistency

    # --- Coalesce Input Tensor ---
    # This is crucial as .indices() and .values() require a coalesced tensor.
    if not tensor_sparse.is_coalesced():
        sparse_coalesced = tensor_sparse.coalesce()
    else:
        sparse_coalesced = tensor_sparse

    # --- Get Sparse Components ---
    original_indices = sparse_coalesced.indices()
    original_values = sparse_coalesced.values()
    
    # --- Create New Sparse Tensor with Target Shape ---
    expanded_tensor = torch.sparse_coo_tensor(
        indices=original_indices,
        values=original_values,
        size=target_shape, # Use the new (m1, m2) target_shape
        dtype=original_values.dtype, # Preserve original values' dtype
        device=tensor_sparse.device  # Preserve original tensor's device
    )
    
    # The new tensor should be coalesced if sparse_coalesced was,
    # because indices and values structure hasn't changed relative to each other.
    # If one wants to be absolutely certain and enforce it:
    # if not expanded_tensor.is_coalesced():
    #     expanded_tensor = expanded_tensor.coalesce()
    # This is usually not necessary if the input `sparse_coalesced` was indeed coalesced
    # and only the size parameter changes in the constructor.

    return expanded_tensor

def expand_sparse_to_dense_tensor_dims_torch(
    tensor_sparse: torch.Tensor,
    target_dense_tensor: torch.Tensor
) -> torch.Tensor:
    """
    将一个 PyTorch COO 稀疏张量扩展到与另一个（通常是稠密的）张量相同的维度。
    这是一个辅助函数，它调用 expand_sparse_to_target_dims_torch。

    参数:
    tensor_sparse (torch.Tensor): 输入的 n1*n2 PyTorch COO 稀疏张量。
    target_dense_tensor (torch.Tensor): 目标张量 (通常是稠密的)，
                                        其形状 (m1, m2) 将被用作扩展目标。

    返回:
    torch.Tensor: 扩展后的 m1*m2 COO 稀疏张量。
    """
    if not isinstance(target_dense_tensor, torch.Tensor):
        raise TypeError("target_dense_tensor 必须是 PyTorch 张量。")
    if target_dense_tensor.ndim != 2:
        raise ValueError(f"target_dense_tensor 必须是二维的，但得到了 {target_dense_tensor.ndim} 维。")
    
    m1_target, m2_target = target_dense_tensor.shape
    return expand_sparse_to_target_dims_torch(tensor_sparse, (m1_target, m2_target))