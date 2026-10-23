import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCDModel

def irt2pl(theta, a, b, *, F=torch):
    """
    2PL IRT模型的项目响应函数

    Parameters:
    -----------
    theta: 潜在能力向量
    a: 题目区分度向量
    b: 题目难度参数
    F: 计算库 (torch或numpy)

    Returns:
    --------
    正确回答概率
    """
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):
    def __init__(self, n_students, n_exercises, latent_dim, a_range):
        super(MIRTNet, self).__init__()
        self.user_num = n_students
        self.item_num = n_exercises
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        self.a_range = a_range

    def forward(self, user, item, q_matrix=None):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(item), dim=-1)
        
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan! The a_range is too large.')
        
        return self.irf(theta, a, b)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


class MIRT(BaseCDModel):
    """
    Multidimensional Item Response Theory Model (MIRT)
    
    Parameters:
    -----------
    n_concepts : int
        Number of latent concepts/dimensions
    n_exercises : int
        Number of exercises/items
    n_students : int
        Number of students/users
    a_range : float or None
        If specified, constrains the discrimination parameters to [0, a_range]
        using sigmoid. If None, softplus is used to ensure positive values.
    """
    def __init__(self, n_students, n_exercises, n_concepts, a_range=None):
        super(MIRT, self).__init__()
        self.mirt_net = MIRTNet(
            n_students=n_students, 
            n_exercises=n_exercises, 
            latent_dim=n_concepts, 
            a_range=a_range
        )
    
    def forward(self, stu_id, exer_id, q_vector=None):
        """
        前向传播计算学生回答正确的概率
        
        Parameters:
        -----------
        stu_id : torch.Tensor
            学生ID张量
        exer_id : torch.Tensor
            题目ID张量
        q_vector : torch.Tensor or None
            Q矩阵向量，在MIRT中通常不使用，保持接口一致性
            
        Returns:
        --------
        torch.Tensor
            预测的回答正确概率
        """
        return self.mirt_net(stu_id, exer_id, q_vector)
    
    def get_knowledge_status(self, stu_id):
        """
        获取学生的知识状态（潜在能力）
        
        Parameters:
        -----------
        stu_id: torch.Tensor or int
            学生ID
            
        Returns:
        --------
        torch.Tensor
            学生的潜在能力向量
        """
        if not isinstance(stu_id, torch.Tensor):
            stu_id = torch.tensor([stu_id])
        return torch.squeeze(self.mirt_net.theta(stu_id), dim=-1)
    
    def get_exercise_params(self, exer_id):
        """
        获取题目参数
        
        Parameters:
        -----------
        exer_id: torch.Tensor or int
            题目ID
            
        Returns:
        --------
        tuple(torch.Tensor, torch.Tensor)
            题目的区分度向量和难度参数
        """
        if not isinstance(exer_id, torch.Tensor):
            exer_id = torch.tensor([exer_id])
        
        a = torch.squeeze(self.mirt_net.a(exer_id), dim=-1)
        if self.mirt_net.a_range is not None:
            a = self.mirt_net.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        
        b = torch.squeeze(self.mirt_net.b(exer_id), dim=-1)
        
        return a, b

    def get_all_knowledge_emb(self):
        """
        返回学生在各个能力维度上的参数
        Returns:
            numpy.ndarray: shape (n_students, n_concepts)
        """
        all_student_ids = torch.arange(self.mirt_net.user_num).to(self.mirt_net.theta.weight.device)
        theta = self.mirt_net.theta(all_student_ids)
        return theta.detach().cpu().numpy()