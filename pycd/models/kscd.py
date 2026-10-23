import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCDModel

class KSCD(BaseCDModel):
    """
    Knowledge-Sensed Cognitive Diagnosis (KSCD) model
    
    As described in the paper: "Knowledge-Sensed Cognitive Diagnosis for 
    Intelligent Education Platforms" (CIKM '22)
    
    Parameters:
    -----------
    n_students : int
        Number of students/users
    n_exercises : int
        Number of exercises/items
    n_concepts : int
        Number of knowledge concepts
    emb_dim : int
        Dimension of embedding vectors
    """
    def __init__(self, n_students, n_exercises, n_concepts, emb_dim=20):
        super(KSCD, self).__init__()
        self.n_students = n_students
        self.n_exercises = n_exercises
        self.n_concepts = n_concepts
        self.emb_dim = emb_dim
        
        # Embedding module
        self.student_emb = nn.Embedding(self.n_students, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.n_exercises, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.n_concepts, self.emb_dim))
        
        # Knowledge-sensed representation module
        self.disc_mlp = nn.Linear(self.emb_dim, 1)  # For exercise discrimination
        
        # Student-exercise interaction module
        self.f_sk = nn.Linear(self.n_concepts + self.emb_dim, self.n_concepts)
        self.f_ek = nn.Linear(self.n_concepts + self.emb_dim, self.n_concepts)
        self.f_se = nn.Linear(self.n_concepts, 1)
        
        # Parameter initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)
    
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
        # Embedding module
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(exer_id)
        
        # Knowledge-sensed representation module
        # Student knowledge mastery vector
        stu_ability = torch.sigmoid(stu_emb @ self.knowledge_emb.T)
        # Exercise knowledge difficulty vector
        diff_emb = torch.sigmoid(exer_emb @ self.knowledge_emb.T)
        # Exercise discrimination
        disc = torch.sigmoid(self.disc_mlp(exer_emb))
        
        # Student-exercise interaction module
        batch_size = stu_id.size(0)
        
        # Replicate student ability and difficulty vectors for each knowledge concept
        stu_emb_expanded = stu_ability.unsqueeze(1).repeat(1, self.n_concepts, 1)
        diff_emb_expanded = diff_emb.unsqueeze(1).repeat(1, self.n_concepts, 1)
        
        # Replicate Q-matrix for each knowledge concept
        q_relevant = q_vector.unsqueeze(2).repeat(1, 1, self.n_concepts)
        
        # Get knowledge embedding representation for each batch item
        knowledge_emb_expanded = self.knowledge_emb.repeat(batch_size, 1).view(batch_size, self.n_concepts, -1)
        
        # Higher-order student knowledge mastery and exercise difficulty
        s_k_concat = torch.sigmoid(self.f_sk(torch.cat([stu_emb_expanded, knowledge_emb_expanded], dim=-1)))
        e_k_concat = torch.sigmoid(self.f_ek(torch.cat([diff_emb_expanded, knowledge_emb_expanded], dim=-1)))
        
        ability_advantage = s_k_concat - e_k_concat
        concept_utility = self.f_se(ability_advantage).squeeze(-1)

        # Knowledge relevance weight (part of formula 10)
        weighted_utility = q_vector * concept_utility

        # Sum over all concepts (part of formula 10)
        utility_sum = torch.sum(weighted_utility, dim=1)

        # Normalize and final activation (remaining part of formula 10)
        # normalized_sum = utility_sum / (disc.view(-1) + 1e-8)
        concept_count = torch.sum(q_vector, dim=1)
        normalized_sum = utility_sum / (concept_count + 1e-8)
        pred = torch.sigmoid(normalized_sum)
                
        return pred
    
    def get_knowledge_mastery(self, stu_id=None):
        """
        Get student's knowledge mastery
        
        Parameters:
        -----------
        stu_id: torch.Tensor or None
            Student ID tensor. If None, return mastery for all students.
            
        Returns:
        --------
        numpy.ndarray
            Student's knowledge mastery matrix of shape (n_students, n_concepts) or
            (batch_size, n_concepts) depending on the input
        """
        if stu_id is None:
            # Return mastery for all students
            return torch.sigmoid(self.student_emb.weight @ self.knowledge_emb.T).detach().cpu().numpy()
        else:
            # Return mastery for specific students
            stu_emb = self.student_emb(stu_id)
            return torch.sigmoid(stu_emb @ self.knowledge_emb.T).detach().cpu().numpy()
    def get_all_knowledge_emb(self):
        """
        Get student's knowledge mastery
        
        Parameters:
        -----------
        stu_id: torch.Tensor or None
            Student ID tensor. If None, return mastery for all students.
            
        Returns:
        --------
        numpy.ndarray
            Student's knowledge mastery matrix of shape (n_students, n_concepts) or
            (batch_size, n_concepts) depending on the input
        """
        # if stu_id is None:
        #     # Return mastery for all students
        #     return torch.sigmoid(self.student_emb.weight @ self.knowledge_emb.T).detach().cpu().numpy()
        # else:
        #     # Return mastery for specific students
        #     stu_emb = self.student_emb(stu_id)
        #     return torch.sigmoid(stu_emb @ self.knowledge_emb.T).detach().cpu().numpy()
        return torch.sigmoid(self.student_emb.weight @ self.knowledge_emb.T).detach().cpu().numpy()