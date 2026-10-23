import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCDModel

class PosLinear(nn.Linear):
    def forward(self, input):
        weight = 2 * F.relu(-self.weight) + self.weight
        return F.linear(input, weight, self.bias)

class NeuralCDM(BaseCDModel):
    """Neural Cognitive Diagnosis Model"""
    def __init__(self, n_concepts, n_exercises, n_students,
                 hidden_dims, dropout):
        super().__init__()
        # 学生因子
        self.student_emb = nn.Embedding(n_students, n_concepts)
        # 练习因子
        self.k_difficulty = nn.Embedding(n_exercises, n_concepts)
        self.e_difficulty = nn.Embedding(n_exercises, 1)
        # 交互网络
        d1, d2 = hidden_dims
        self.fc1 = PosLinear(n_concepts, d1)
        self.drop1 = nn.Dropout(dropout[0])
        self.fc2 = PosLinear(d1, d2)
        self.drop2 = nn.Dropout(dropout[1])
        self.fc3 = PosLinear(d2, 1)

        # 参数初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, q_vector):
        # 学生掌握度
        hs = torch.sigmoid(self.student_emb(stu_id))
        # 练习难度与区分度
        hd = torch.sigmoid(self.k_difficulty(exer_id))
        disc = torch.sigmoid(self.e_difficulty(exer_id))
        # 交互层
        x = disc * (hs - hd) * q_vector
        x = self.drop1(torch.sigmoid(self.fc1(x)))
        x = self.drop2(torch.sigmoid(self.fc2(x)))
        y = torch.sigmoid(self.fc3(x)).view(-1)
        return y
    def get_all_knowledge_emb(self):
        return torch.sigmoid(self.student_emb.weight).detach().cpu().numpy()