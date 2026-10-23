import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.autograd as autograd
import torch.nn.functional as F

from .base import BaseCDModel

class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        """
        DINA Network implementation
        
        Args:
            user_num: Number of students
            item_num: Number of exercises/items
            hidden_dim: Dimension of knowledge state (should match number of knowledge components)
            max_slip: Maximum slip parameter constraint
            max_guess: Maximum guess parameter constraint
        """
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.hidden_dim = hidden_dim
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        """
        Forward pass for DINA model
        
        Args:
            user: User/student IDs
            item: Item/exercise IDs
            knowledge: Knowledge relevance vectors from Q-matrix
            
        Returns:
            Probability of correct response
        """
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
            
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINANet(DINANet):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
            
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)


class DINA(BaseCDModel):
    """
    DINA (Deterministic Inputs, Noisy And-Gate) Model
    
    Parameters:
    -----------
    user_num : int
        Number of students/users
    item_num : int
        Number of exercises/items
    hidden_dim : int
        Dimension of knowledge state (theta) - should match number of knowledge concepts
    ste : bool
        Whether to use Straight-Through Estimator
    """
    def __init__(self, user_num, item_num, hidden_dim, concept_dim=None, ste=False):
        super(DINA, self).__init__()
        # Ensure hidden_dim matches concept_dim for theoretical soundness
        self.hidden_dim = concept_dim if concept_dim is not None else hidden_dim
        
        if ste:
            self.dina_net = STEDINANet(user_num, item_num, self.hidden_dim)
        else:
            self.dina_net = DINANet(user_num, item_num, self.hidden_dim)
    
    def forward(self, stu_id, exer_id, q_vector):
        """
        Forward pass to calculate probability of correct response
        
        Parameters:
        -----------
        stu_id : torch.Tensor
            Student ID tensor
        exer_id : torch.Tensor
            Exercise ID tensor
        q_vector : torch.Tensor
            Knowledge relevance vector from Q-matrix
            
        Returns:
        --------
        torch.Tensor
            Predicted probability of correct response
        """
        return self.dina_net(stu_id, exer_id, q_vector)
    
    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dina_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
    
    def get_all_knowledge_emb(self):
        """
        返回学生在各知识点上的掌握状态
        Returns:
            numpy.ndarray: shape (n_students, n_concepts), 值为0或1
        """
        all_student_ids = torch.arange(self.dina_net._user_num).to(self.dina_net.theta.weight.device)
        theta = self.dina_net.theta(all_student_ids)
        
        if hasattr(self.dina_net, 'sign'):  # STE版本
            knowledge_state = (theta > 0).float()
        else:
            knowledge_state = (torch.sigmoid(theta) > 0.5).float()
        
        return knowledge_state.detach().cpu().numpy()