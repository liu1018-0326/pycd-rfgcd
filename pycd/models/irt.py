import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCDModel
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def irt3pl(theta, a, b, c, *, F=torch):
    """
    3PL IRT model's item response function

    Parameters:
    -----------
    theta: latent ability parameter (1D)
    a: item discrimination parameter
    b: item difficulty parameter
    c: item guessing parameter
    F: computation library (torch or numpy)

    Returns:
    --------
    probability of correct response
    """
    return c + (1 - c) / (1 + F.exp(-a * (theta - b)))


class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range=None, a_range=None, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        self.value_range = value_range
        self.a_range = a_range

    def forward(self, user, item, q_matrix=None):
        """
        Forward pass to calculate probability of correct response
        
        Parameters:
        -----------
        user : torch.Tensor
            User ID tensor
        item : torch.Tensor
            Item ID tensor
        q_matrix : torch.Tensor or None
            Q-matrix (not used in IRT but kept for interface consistency)
            
        Returns:
        --------
        torch.Tensor
            Predicted probability of correct response
        """
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
            
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan! The value_range or a_range is too large.')
            
        return self.irf(theta, a, b, c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)


class IRT(BaseCDModel):
    """
    Item Response Theory Model (IRT)
    
    Parameters:
    -----------
    n_students : int
        Number of students/users
    n_exercises : int
        Number of exercises/items
    value_range : float or None
        If specified, constrains the theta and b parameters to [-value_range/2, value_range/2]
        using sigmoid transformation. If None, no constraints are applied.
    a_range : float or None
        If specified, constrains the discrimination parameters to [0, a_range]
        using sigmoid. If None, softplus is used to ensure positive values.
    """
    def __init__(self, n_students, n_exercises, value_range=None, a_range=None):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(n_students, n_exercises, value_range, a_range)
    
    def forward(self, stu_id, exer_id, q_vector=None):
        """
        Forward pass to calculate probability of correct response
        
        Parameters:
        -----------
        stu_id : torch.Tensor
            Student ID tensor
        exer_id : torch.Tensor
            Exercise ID tensor
        q_vector : torch.Tensor or None
            Q-matrix vector, not used in IRT but kept for interface consistency
            
        Returns:
        --------
        torch.Tensor
            Predicted probability of correct response
        """
        return self.irt_net(stu_id, exer_id, q_vector)
    
    def get_student_ability(self, stu_id):
        """
        Get student's latent ability parameter
        
        Parameters:
        -----------
        stu_id: torch.Tensor or int
            Student ID
            
        Returns:
        --------
        torch.Tensor
            Student's latent ability parameter
        """
        if not isinstance(stu_id, torch.Tensor):
            stu_id = torch.tensor([stu_id])
        theta = torch.squeeze(self.irt_net.theta(stu_id), dim=-1)
        
        if self.irt_net.value_range is not None:
            theta = self.irt_net.value_range * (torch.sigmoid(theta) - 0.5)
            
        return theta
    
    def get_exercise_params(self, exer_id):
        """
        Get exercise parameters
        
        Parameters:
        -----------
        exer_id: torch.Tensor or int
            Exercise ID
            
        Returns:
        --------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor)
            Exercise's discrimination, difficulty, and guessing parameters
        """
        if not isinstance(exer_id, torch.Tensor):
            exer_id = torch.tensor([exer_id])
        
        a = torch.squeeze(self.irt_net.a(exer_id), dim=-1)
        b = torch.squeeze(self.irt_net.b(exer_id), dim=-1)
        c = torch.squeeze(self.irt_net.c(exer_id), dim=-1)
        
        if self.irt_net.value_range is not None:
            b = self.irt_net.value_range * (torch.sigmoid(b) - 0.5)
            
        if self.irt_net.a_range is not None:
            a = self.irt_net.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
            
        c = torch.sigmoid(c)
        
        return a, b, c

    def get_all_knowledge_emb(self):
        """
        IRT模型不支持知识点级别的能力建模，返回None
        这样DOA计算函数可以优雅地处理这种情况
        """
        import warnings
        warnings.warn(
            "IRT model does not support knowledge-specific ability modeling. "
            "DOA calculation will return NaN.",
            UserWarning,
            stacklevel=2
        )
        return None