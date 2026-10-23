import json
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import scipy.sparse as sp
import numbers
from pycd.models import BaseCDModel
from torch.utils.data import DataLoader
from scipy.special import expit
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.cluster import KMeans
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed

'''
DOA HI CI 待对比迁移
'''
#from DOA import DOA
#from homogeneity import cosine_similarity, euclidean_similarity


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, device="cuda"):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim  #自编码器：输入维度 = 输出维度
        self.hidden_dims = hidden_dims.copy()  # 创建副本避免修改原始列表
        self.hidden_dims.append(latent_dim)  #隐藏层（可能多个层） #【a,b,c】
        self.dims_list = (self.hidden_dims + self.hidden_dims[:-1][::-1])  # mirrored structure 【a,b,c】--> 【a,b,c,b,a】
        self.n_layers = len(self.dims_list)  #网络总层数
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.device = device

        # Validation check
        assert self.n_layers % 2 > 0 #确保网络层数为奇数层
        assert self.dims_list[self.n_layers // 2] == self.latent_dim #//整除，检查中间那一层

        # Encoder Network
        layers = OrderedDict()  #有序字典layers，顺序存储每一层
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:  #第一层，输入层
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64),
                        'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim, dtype=torch.float64),  #前一层-->当前层
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx], dtype=torch.float64)
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]  #翻转操作，【a,b,c】--> 【c,b,a】
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:  #最后一层，输出层
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim, dtype=torch.float64),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1], dtype=torch.float64),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1], dtype=torch.float64)
                    }
                )
        self.decoder = nn.Sequential(layers)
        
        # 将模型移至指定设备
        self.to(self.device)

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output  #取中间值
        return self.decoder(output)  #取最终值


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))   #存放每个样本到该类cluster的距离
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))  #计算欧几里得距离
    return dis_mat


class BatchKMeans(object):
    #初始化参数
    def __init__(self, latent_dim, n_clusters, n_jobs):
        self.n_features = latent_dim  #聚类中心的维度（与AutoEncoder得到的latent_dim维度相同）
        self.n_clusters = n_clusters  #聚类中心数
        self.clusters = np.zeros((self.n_clusters, self.n_features))   #【n_clusters ✖️ n_features】
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate #每个聚类中心被分配过几次，初始值为100
        self.n_jobs = n_jobs   #并行线程数量
    #计算样本到所有聚类中心的欧几里得距离
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i]) #每个样本到第i个聚类中心的距离
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat) #横向堆叠-->【n_samples, n_clusters】每一行表示一个样本对所有聚类中心的距离。

        return dis_mat
    #初始化聚类中心
    def init_cluster(self, X):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters, n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters
        #self.initial_labels = model.labels_  # 保存样本初始归属簇
    #更新聚类中心
    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """  #增量式（在线）更新
        n_samples = X.shape[0]  #该batch被分配给cluster_idx类的所有样本
        for i in range(n_samples):
            self.count[cluster_idx] += 1  #初始时是100
            eta = 1.0 / self.count[cluster_idx]   #计算该聚类中心当前的学习率，被分配次数越多，学习率越小
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                               eta * X[i])   #根据单个样本更新聚类中心
            self.clusters[cluster_idx] = updated_cluster
   #返回每个样本最近的聚类中心
    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1) #按行找最小值
    #返回每个样本最近的belong个聚类中心
    def assign_group(self, X, belong):
        dis_mat = self._compute_dist(X)
        return np.argsort(dis_mat, axis=1)[:, :belong] #按行找前belong个最小值，允许一个学生属于多个类


class DeepClusteringNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, beta=1, lamda=1,
                 pretrained=True, lr=0.0001, device="cuda", n_jobs=-1):
        super(DeepClusteringNet, self).__init__()
        self.beta = beta  # coefficient of the clustering term #聚类损失的权重
        self.lamda = lamda  # coefficient of the reconstruction term  #AutoEncoder重构损失的权重
        self.device = device
        self.pretrained = pretrained
        self.n_clusters = n_clusters

        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))

        if len(hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')

        self.kmeans = BatchKMeans(latent_dim, n_clusters, n_jobs)
        self.autoencoder = AutoEncoder(input_dim, hidden_dims, latent_dim, n_clusters, device=self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr,
                                          weight_decay=5e-4)
        
        # 将模型移至指定设备
        self.to(self.device)

    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]   #当前batch的样本数
        rec_X = self.autoencoder(X)   #
        latent_X = self.autoencoder(X, latent=True)  #True

        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)

        # Regularization term on clustering
        dist_loss = torch.tensor(0., device=self.device, dtype=torch.float64)   #初始时，聚类损失为0
        clusters = torch.tensor(self.kmeans.clusters, dtype=torch.float64, device=self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1)) #计算样本到聚类中心的欧几里得距离
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss) #累加所以样本损失 1/2* beta *||x_i - c_k||^2

        return (rec_loss + dist_loss,
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())

    def pretrain(self, train_loader, epoch=100):
        if not self.pretrained:
            return
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))

        rec_loss_list = [] #记录每一轮的重构损失

        self.train()
        for e in tqdm(range(epoch), "gain feature"):
            for data in train_loader:
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)

                rec_loss_list.append(loss.detach().cpu().numpy())

                self.optimizer.zero_grad() #梯度清0
                loss.backward()  #反向传播
                self.optimizer.step()  #更新参数
        self.eval()

        # Initialize clusters in self.kmeans after pre-training
        batch_X = [] #【num_students✖️latent_dim】
        for data in train_loader:
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)  #初始化聚类中心

        return rec_loss_list

    def fit(self, train_loader, epoch=50):
        for e in tqdm(range(epoch), "clustering"):
            self.train()
            for data in train_loader:
                batch_size = data.size()[0]  #该batch样本数量
                data = data.view(batch_size, -1).to(self.device)   #--> 【batch_size✖️input_dim】

                # Get the latent features
                with torch.no_grad():  #不需要反向传播
                    latent_X = self.autoencoder(data, latent=True)
                    latent_X = latent_X.cpu().numpy()  #转成numpy给Kmeans处理

                # [Step-1] Update the assignment results  #得到每个样本的聚类类别
                cluster_id = self.kmeans.update_assign(latent_X)

                # [Step-2] Update clusters in bath Kmeans
                elem_count = np.bincount(cluster_id,
                                         minlength=self.n_clusters)  #统计每个聚类中心包含了多少个样本
                for k in range(self.n_clusters):
                    # avoid empty slicing
                    if elem_count[k] == 0:
                        continue
                    self.kmeans.update_cluster(latent_X[cluster_id == k], k)   #更新聚类中心

                # [Step-3] Update the network parameters
                loss, rec_loss, dist_loss = self._loss(data, cluster_id)   #计算损失
                self.optimizer.zero_grad()  #梯度清0
                loss.backward()  #反向传播
                self.optimizer.step()  #参数更新

    def gain_clusters(self, train_loader, belong):
        clusters = []  #累计所有batch中样本的聚类分配结果
        for data in train_loader:
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)

            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()

            cluster_id = self.kmeans.assign_group(latent_X, belong)
            clusters.append(cluster_id)
        group_ids = np.vstack(clusters)
        return group_ids


class Hypergraph:
    def __init__(self, H: np.ndarray):
        self.H = H
        # avoid zero
        self.Dv = np.count_nonzero(H, axis=1) + 1  #节点度向量：每个节点参与了多少超边
        self.De = np.count_nonzero(H, axis=0) + 1  #超边度向量：每条超边连接多少个节点

    def to_tensor_nadj(self):
        coo = sp.coo_matrix(self.H @ np.diag(1 / self.De) @ self.H.T @ np.diag(1 / self.Dv)) #超图邻接矩阵归一化：A = H * D_e^{-1} * H^T *D_v^{-1}
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        values = torch.from_numpy(coo.data).double()  # 确保使用float64类型
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), dtype=torch.float64).coalesce()


#约束神经网络层的权重非负-->正权重不变，负权重置0
class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class HSCD_Net(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num, feature_dim, emb_dim,
                 student_adj, exercise_adj, knowledge_adj, device, layers=3, leaky=0.8):
        super(HSCD_Net, self).__init__()

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim

        self.device = device
        self.layers = layers
        self.leaky = leaky

        assert student_adj is not None and exercise_adj is not None and knowledge_adj is not None
        self.register_buffer('student_adj',   student_adj)
        self.register_buffer('exercise_adj',  exercise_adj)
        self.register_buffer('knowledge_adj', knowledge_adj)

        self.student_emb = nn.Embedding(student_num, emb_dim, dtype=torch.float64)
        self.exercise_emb = nn.Embedding(exercise_num, emb_dim, dtype=torch.float64)
        self.knowledge_emb = nn.Embedding(knowledge_num, emb_dim, dtype=torch.float64)

        self.student_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.exercise_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.knowledge_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.exercise_emb2discrimination = nn.Linear(emb_dim, 1, dtype=torch.float64)

        self.clipper = NoneNegClipper()

        self.state2response = nn.Sequential(
            nn.Linear(knowledge_num, 512, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(128, 1, dtype=torch.float64),
            nn.Sigmoid()
        )

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        
        # 将模型移至指定设备
        self.to(self.device)

    def convolution(self, embedding, adj):
        adj = adj.to(embedding.weight.device)
        all_emb = embedding.weight 
        final = [all_emb]
        for i in range(self.layers):
            all_emb = torch.sparse.mm(adj, all_emb) + 0.8 * all_emb
            final.append(all_emb)
        final_emb = torch.mean(torch.stack(final, dim=1), dim=1, dtype=torch.float64)
        return final_emb

    def forward(self, student_id, exercise_id, knowledge):

        student_id  = student_id.to(self.student_emb.weight.device)
        exercise_id = exercise_id.to(self.exercise_emb.weight.device)
        knowledge = knowledge.to(self.knowledge_emb.weight.device)

        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_exercise_emb = self.convolution(self.exercise_emb, self.exercise_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        batch_student = f.embedding(student_id, convolved_student_emb)
        batch_exercise = f.embedding(exercise_id, convolved_exercise_emb)

        student_feature = f.leaky_relu(self.student_emb2feature(batch_student), negative_slope=self.leaky)
        exercise_feature = f.leaky_relu(self.exercise_emb2feature(batch_exercise), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)
        discrimination = torch.sigmoid(self.exercise_emb2discrimination(batch_exercise))

        state = discrimination * (student_feature @ knowledge_feature.T
                                  - exercise_feature @ knowledge_feature.T) * knowledge

        state = self.state2response(state)
        return state.view(-1)

    def apply_clipper(self):
        for layer in self.state2response:
            if isinstance(layer, nn.Linear):
                layer.apply(self.clipper)

    def get_proficiency_level(self):
        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        student_feature = f.leaky_relu(self.student_emb2feature(convolved_student_emb), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)

        return torch.sigmoid(student_feature @ knowledge_feature.T).detach().cpu().numpy()

    def get_exercise_level(self):
        convolved_exercise_emb = self.convolution(self.exercise_emb, self.exercise_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        exercise_feature = f.leaky_relu(self.exercise_emb2feature(convolved_exercise_emb), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)

        return torch.sigmoid(exercise_feature @ knowledge_feature.T).detach().cpu().numpy()

    def get_knowledge_feature(self):
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)
        return knowledge_feature.detach().cpu().numpy()
    def get_all_knowledge_emb(self):
        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        student_feature = f.leaky_relu(self.student_emb2feature(convolved_student_emb), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)

        return torch.sigmoid(student_feature @ knowledge_feature.T).detach().cpu().numpy()


class HyperCDM(BaseCDModel):
    def __init__(self, student_num, exercise_num, knowledge_num,
                 feature_dim, emb_dim, layers, device='cuda'):
        super(HyperCDM, self).__init__()

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.feature_dim = feature_dim   #全连接层映射后的维度
        self.emb_dim = emb_dim   #s、e、k的初始嵌入维度
        self.layers = layers   #图卷积层数
        self.device = device

        self.net = None  # 模型结构稍后在 hyper_build 构建
        
        # 将模型移至指定设备
        self.to(self.device)

    def forward(self, student_id, exercise_id, q_vector):
        return self.net(student_id.to(self.device),
                        exercise_id.to(self.device),
                        q_vector.to(self.device))

    def loss(self, preds, labels):
        """
        BCE loss，确保 pred 和 label 类型一致。
        """
        labels = labels.to(dtype=preds.dtype)  # 自动转换成 preds 的 dtype
        return nn.functional.binary_cross_entropy(preds, labels)
    
    def get_all_knowledge_emb(self):
        return self.net.get_all_knowledge_emb()

    def hyper_build(self, response_logs, q_matrix, r_matrix):
        """
        构建三类超图：学生图、题目图、知识点图，并初始化网络结构 self.net
        """
        print("Construct student hypergraph")
        # 确保使用float64类型
        X = torch.tensor(r_matrix, dtype=torch.float64)
        n_clusters = int(self.student_num * 0.02)

        student_loader = DataLoader(dataset=X, batch_size=256, shuffle=False)

        clf = DeepClusteringNet(
            input_dim=self.exercise_num,
            hidden_dims=[512, 256, 128],
            latent_dim=64,
            n_clusters=n_clusters,
            device=self.device
        )
        clf.pretrain(student_loader)
        clf.fit(student_loader)

        groups = clf.gain_clusters(student_loader, n_clusters // 2)
        H_student = np.zeros((self.student_num, n_clusters))
        for i in range(H_student.shape[0]):
            H_student[i, groups[i]] = 1
        H_student = H_student[:, np.count_nonzero(H_student, axis=0) >= 2]
        self.student_hyper = Hypergraph(H_student)

        print("Construct exercise hypergraph")
        H_exer = q_matrix.copy()
        H_exer = H_exer[:, np.count_nonzero(H_exer, axis=0) >= 2]
        self.exercise_hyper = Hypergraph(H_exer)

        print("Construct knowledge concept hypergraph")
        H_know = q_matrix.T.copy()
        H_know = H_know[:, np.count_nonzero(H_know, axis=0) >= 2]
        self.knowledge_hyper = Hypergraph(H_know)

        # 初始化最终网络结构
        self.net = HSCD_Net(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            feature_dim=self.feature_dim,
            emb_dim=self.emb_dim,
            layers=self.layers,
            student_adj=self.student_hyper.to_tensor_nadj().to(self.device),
            exercise_adj=self.exercise_hyper.to_tensor_nadj().to(self.device),
            knowledge_adj=self.knowledge_hyper.to_tensor_nadj().to(self.device),
            device=self.device
        )
    
    def train_model(self, train_loader, valid_loader=None, epochs=4, lr=0.0001):
        if self.net is None:
            raise RuntimeError("Use hyper_build() method first")
        
        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0005)
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"Total number of parameters of HyperCDM: {total_params}")
        
        for epoch_i in range(epochs):
            self.net.train()
            epoch_losses = []
            
            for batch_data in tqdm(train_loader, f"Epoch {epoch_i + 1}"):
                student_id, exercise_id, knowledge, y = batch_data
                student_id = student_id.to(self.device)
                exercise_id = exercise_id.to(self.device)
                knowledge = knowledge.to(self.device)
                y = y.to(self.device)
                
                pred_y = self.net.forward(student_id, exercise_id, knowledge)
                bce_loss = bce_loss_function(pred_y, y)
                
                optimizer.zero_grad()
                bce_loss.backward()
                optimizer.step()
                self.net.apply_clipper()
                
                epoch_losses.append(bce_loss.mean().item())
            
            print(f"[Epoch {epoch_i + 1}] average loss: {float(np.mean(epoch_losses)):.6f}")
            
            if valid_loader is not None:
                metrics = self.evaluate(valid_loader)
                print(f"Validation metrics: {metrics}")
    
    def evaluate(self, test_loader):
        self.net.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Evaluating"):
                student_id, exercise_id, knowledge, y = batch_data
                student_id = student_id.to(self.device)
                exercise_id = exercise_id.to(self.device)
                knowledge = knowledge.to(self.device)
                
                pred_y = self.net.forward(student_id, exercise_id, knowledge)
                y_pred.extend(pred_y.cpu().tolist())
                y_true.extend(y.cpu().tolist())
        
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        f1 = f1_score(y_true, np.array(y_pred) >= 0.5)
        
        return {"auc": auc, "acc": acc, "rmse": rmse, "f1": f1}


def extract_response_logs(dataset):
    """
    从 CDMDataset 提取 HyperCDM 构图所需的响应矩阵。
    返回一个字典列表，每个 dict 是一个答题记录。
    """
    logs = []
    for u_idx, q_idx, q_vector, label in dataset:
        logs.append({
            "user_id": int(u_idx.item()),
            "item_id": int(q_idx.item()),
            "score": int(label.item())
        })
    return logs


def build_r_matrix(dataset, num_users, num_exercises):
    """
    从 CDMDataset 构造响应矩阵 R (user × item)，答对为 1，答错为 0。
    没做题的留空（值为 -1）
    """
    R = -1 * np.ones((num_users, num_exercises), dtype=np.int8)
    for u_idx, q_idx, _, label in dataset:
        R[u_idx.item(), q_idx.item()] = int(label.item())
    return R