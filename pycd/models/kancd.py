# pycd/models/kancd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCDModel

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class KaNCD(BaseCDModel):
    """Knowledge and Neural Cognitive Diagnosis Model"""
    def __init__(self, n_concepts, n_exercises, n_students,
                 emb_dim=20, mf_type='gmf',
                 hidden_dims=(256, 128), dropout=(0.5, 0.5)):
        
        assert mf_type in ['mf', 'gmf', 'ncf1', 'ncf2'], \
        f"mf_type must be one of ['mf', 'gmf', 'ncf1', 'ncf2'], but got {mf_type}"
        
        super().__init__()
        self.n_concepts = n_concepts
        self.n_exercises = n_exercises
        self.n_students = n_students
        self.emb_dim = emb_dim
        self.mf_type = mf_type
        self.prednet_len1, self.prednet_len2 = hidden_dims

        # 嵌入层 - 保持与原始实现相同的命名
        self.student_emb = nn.Embedding(n_students, emb_dim)
        self.exercise_emb = nn.Embedding(n_exercises, emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(n_concepts, emb_dim))
        self.e_discrimination = nn.Embedding(n_exercises, 1)

        # 预测网络 
        self.prednet_full1 = PosLinear(n_concepts, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout[0])
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout[1])
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # MF类型特定的层
        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(emb_dim, 1)
            self.stat_full = nn.Linear(emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * emb_dim, 1)
            self.stat_full = nn.Linear(2 * emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * emb_dim, emb_dim)
            self.k_diff_full2 = nn.Linear(emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * emb_dim, emb_dim)
            self.stat_full2 = nn.Linear(emb_dim, 1)

        # 参数初始化 - 与原始实现保持一致
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, exer_id, q_vector):
        # 获取嵌入
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(exer_id)
        
        # 获取知识状态 (knowledge proficiency)
        batch, dim = stu_emb.size()
        stu_emb_expanded = stu_emb.view(batch, 1, dim).repeat(1, self.n_concepts, 1)
        knowledge_emb_expanded = self.knowledge_emb.repeat(batch, 1).view(batch, self.n_concepts, -1)
        
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb_expanded * knowledge_emb_expanded).sum(dim=-1, keepdim=False))
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb_expanded * knowledge_emb_expanded)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb_expanded, knowledge_emb_expanded), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb_expanded, knowledge_emb_expanded), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
            
        # 获取知识难度
        batch, dim = exer_emb.size()
        exer_emb_expanded = exer_emb.view(batch, 1, dim).repeat(1, self.n_concepts, 1)
        
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb_expanded * knowledge_emb_expanded).sum(dim=-1, keepdim=False))
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb_expanded * knowledge_emb_expanded)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb_expanded, knowledge_emb_expanded), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb_expanded, knowledge_emb_expanded), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
            
        # 获取练习区分度
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))
        
        # 预测网络
        input_x = e_discrimination * (stat_emb - k_difficulty) * q_vector
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        
        return output.view(-1)
    def get_all_knowledge_emb(self):
        stu_emb = self.student_emb.weight
        
        
        # 获取知识状态 (knowledge proficiency)
        batch, dim = stu_emb.size()
        stu_emb_expanded = stu_emb.view(batch, 1, dim).repeat(1, self.n_concepts, 1)
        knowledge_emb_expanded = self.knowledge_emb.repeat(batch, 1).view(batch, self.n_concepts, -1)
        
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb_expanded * knowledge_emb_expanded).sum(dim=-1, keepdim=False))
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb_expanded * knowledge_emb_expanded)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb_expanded, knowledge_emb_expanded), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb_expanded, knowledge_emb_expanded), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        return stat_emb