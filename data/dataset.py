# PYCD/data/dataset.py
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_question_mapping(mapping_path: str) -> dict:
    """
    从 JSON 文件加载题目 ID 映射关系。
    JSON 格式示例：{"questions": {"原始题目ID": 映射后索引, ...}}
    返回一个 dict，将原始题目ID（str）映射为新的连续索引（int）。
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(orig): int(mapped) for orig, mapped in data['questions'].items()}


def load_q_matrix(
    q_matrix_path: str,
    question2idx: dict = None,
    num_exercises: int = None,
    num_concepts: int = None,
    auto_compress: bool = False  # 添加参数控制是否自动压缩
) -> np.ndarray:
    """
    从CSV文件加载Q矩阵
    
    CSV格式:
    question_id,concept_ids
    0,"[9]"
    1,"[77, 88, 21]"
    
    Parameters:
    -----------
    q_matrix_path: str
        Q矩阵CSV文件路径
    question2idx: dict
        题目ID映射，如果已经在CSV中使用了映射后的ID则可为None
    num_exercises: int
        题目数量，如果为None则自动计算
    num_concepts: int
        概念数量，如果为None则自动计算
    auto_compress: bool
        是否自动压缩未使用的列，默认False
        
    Returns:
    --------
    np.ndarray
        形状为(num_exercises, num_concepts)的多热Q矩阵
    """
    
    # 读取CSV文件
    df = pd.read_csv(q_matrix_path)
    
    # 确保列名正确
    if 'question_id' not in df.columns or 'concept_ids' not in df.columns:
        raise ValueError(f"CSV文件必须包含question_id和concept_ids列，实际列名: {df.columns}")
    
    # 如果没有指定题目数量，从数据中计算
    if num_exercises is None:
        if question2idx is not None:
            num_exercises = len(question2idx)
        else:
            num_exercises = df['question_id'].max() + 1
    
    # 解析concept_ids列并找出最大概念ID
    all_concepts = set()
    concept_usage = {}  # 跟踪每个概念ID的使用次数
    rows = []
    
    for _, row in df.iterrows():
        item_id = row['question_id']
        # 解析概念ID字符串为列表
        try:
            # 尝试多种格式解析
            concept_str = row['concept_ids']
            
            # 如果是字符串形式，转换为Python对象
            if isinstance(concept_str, str):
                # 尝试多种可能的格式
                try:
                    # 标准JSON格式
                    concept_ids = json.loads(concept_str.replace("'", '"'))
                except:
                    try:
                        # Python列表字符串
                        concept_ids = eval(concept_str)
                    except:
                        # 简单逗号分隔的数字
                        concept_ids = [int(c.strip()) for c in concept_str.strip('[]').split(',') if c.strip()]
            else:
                # 已经是列表或其他对象
                concept_ids = concept_str
            
            # 确保concept_ids是列表
            if not isinstance(concept_ids, list):
                concept_ids = [concept_ids]
            
            # 更新所有概念集合
            all_concepts.update(concept_ids)
            
            # 更新概念使用计数
            for cid in concept_ids:
                if cid in concept_usage:
                    concept_usage[cid] += 1
                else:
                    concept_usage[cid] = 1
            
            # 如果存在题目ID映射，应用它
            if question2idx is not None:
                if str(item_id) in question2idx:
                    mapped_id = question2idx[str(item_id)]
                else:
                    # 跳过未映射的题目
                    print(f"警告: 题目ID {item_id} 未在映射中找到，将被跳过")
                    continue
            else:
                mapped_id = item_id
            
            rows.append((mapped_id, concept_ids))
        except Exception as e:
            print(f"解析行错误 {row}: {e}")
    
    # 如果未指定概念数量，使用指定值或找到的最大概念ID + 1
    if num_concepts is None:
        if all_concepts:
            num_concepts = max(all_concepts) + 1
        else:
            num_concepts = 1
            print("警告: 未找到任何概念ID")
    
    
    # 创建并填充多热Q矩阵
    q_matrix = np.zeros((num_exercises, num_concepts), dtype=np.float32)
    
    for item_id, concept_ids in rows:
        for concept_id in concept_ids:
            if concept_id < num_concepts:
                q_matrix[item_id, concept_id] = 1.0
    
    # 检查Q矩阵的使用情况
    used_concepts = np.sum(q_matrix, axis=0) > 0
    actual_used_count = np.sum(used_concepts)
    
    # 打印未使用的概念ID列表
    unused_concepts = np.where(np.sum(q_matrix, axis=0) == 0)[0]
    
    # 如果需要并且概念数量明显不合理，返回压缩后的矩阵
    if auto_compress and actual_used_count < num_concepts * 0.8:  # 如果使用的概念不到80%
        # 只保留非零列
        nonzero_cols = np.where(used_concepts)[0]
        compressed_q_matrix = q_matrix[:, nonzero_cols]
        # print(f"压缩后的Q矩阵形状: {compressed_q_matrix.shape}")
        return compressed_q_matrix
    
    return q_matrix

class RCDDataset(Dataset):
    """
    RCD任务数据集。
    返回 (student_idx, exercise_idx, q_vector, label)。
    """
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        user_id = data['user_id']
        exer_id = data['exer_id']
        q_vector = torch.tensor(data['q_vector'], dtype=torch.float)
        label = data['correct']
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(exer_id, dtype=torch.long),
            q_vector,
            torch.tensor(label, dtype=torch.float)
        )

# class CDMDataset(Dataset):
#     """
#     认知诊断任务数据集。
#     返回 (student_idx, exercise_idx, q_vector, label)。
#     """
#     def __init__(
#         self,
#         csv_path: str,
#         question2idx: dict,
#         q_matrix: np.ndarray,
#         user2idx: dict = None,
#         normalize_label: bool = True
#     ):
#         df = pd.read_csv(csv_path)
#         # 映射题目 ID
#         df['q_idx'] = df['question_id'].astype(int)
#         # 映射用户 ID（如果提供映射）
#         if user2idx is not None:
#             df['u_idx'] = df['user_id'].astype(int)
#         else:
#             df['u_idx'] = df['user_id'].astype(int)
#         df = df.dropna(subset=['q_idx', 'u_idx', 'correct'])
#         df['q_idx'] = df['q_idx'].astype(int)
#         df['u_idx'] = df['u_idx'].astype(int)
#         self.user_idxs = df['u_idx'].values
#         self.q_idxs = df['q_idx'].values
#         labels = df['correct'].astype(np.float32).values
#         if normalize_label:
#             max_label = labels.max()
#             if max_label > 1.0:
#                 labels = labels / max_label
#         self.labels = labels
#         self.q_matrix = torch.from_numpy(q_matrix)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         u = self.user_idxs[idx]
#         q = self.q_idxs[idx]
#         label = self.labels[idx]
#         q_vector = self.q_matrix[q]
#         return (
#             torch.tensor(u, dtype=torch.long),
#             torch.tensor(q, dtype=torch.long),
#             q_vector,
#             torch.tensor(label, dtype=torch.float)
#         )
class CDMDataset(Dataset):
    """
    认知诊断任务数据集。
    返回 (student_idx, exercise_idx, q_vector, label)。
    支持交叉验证。
    """
    def __init__(
        self,
        csv_path: str,
        question2idx: dict,
        q_matrix: np.ndarray,
        user2idx: dict = None,
        normalize_label: bool = True,
        fold_mode: str = None,  # 'train', 'valid', 或 None
        fold: int = None,       # 当前fold编号
        is_test: bool = False   # 是否为测试集
    ):
        df = pd.read_csv(csv_path)
        
        # 处理交叉验证数据筛选
        if not is_test and fold_mode and fold is not None and 'fold' in df.columns:
            if fold_mode == 'valid':
                # 验证集：选取指定fold的数据
                df = df[df['fold'] == fold].copy()
            elif fold_mode == 'train':
                # 训练集：选取除指定fold外的所有数据
                df = df[df['fold'] != fold].copy()
            
            if len(df) == 0:
                raise ValueError(f"筛选后数据集为空! fold_mode={fold_mode}, fold={fold}")
        
        # 映射题目 ID
        df['q_idx'] = df['question_id'].astype(int)
        # 映射用户 ID（如果提供映射）
        if user2idx is not None:
            df['u_idx'] = df['user_id'].astype(int)
        else:
            df['u_idx'] = df['user_id'].astype(int)
        df = df.dropna(subset=['q_idx', 'u_idx', 'correct'])
        df['q_idx'] = df['q_idx'].astype(int)
        df['u_idx'] = df['u_idx'].astype(int)
        self.user_idxs = df['u_idx'].values
        self.q_idxs = df['q_idx'].values
        labels = df['correct'].astype(np.float32).values
        if normalize_label:
            max_label = labels.max()
            if max_label > 1.0:
                labels = labels / max_label
        self.labels = labels
        self.q_matrix = torch.from_numpy(q_matrix)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u = self.user_idxs[idx]
        q = self.q_idxs[idx]
        label = self.labels[idx]
        q_vector = self.q_matrix[q]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(q, dtype=torch.long),
            q_vector,
            torch.tensor(label, dtype=torch.float)
        )
# def get_datasets(
#     mapping_path: str,
#     q_matrix_path: str,
#     item_count: int,
#     concept_count: int,
#     train_csv: str,
#     valid_csv: str,
#     test_csv: str
# ):
#     """
#     一次性加载映射、Q矩阵，并返回 train/valid/test 数据集。
#     """
#     question2idx = load_question_mapping(mapping_path)
#     q_matrix = load_q_matrix(q_matrix_path, None,
#                            num_exercises=len(question2idx),
#                            num_concepts=None)
#     train_ds = CDMDataset(train_csv, question2idx, q_matrix)
#     valid_ds = CDMDataset(valid_csv, question2idx, q_matrix)
#     test_ds  = CDMDataset(test_csv,  question2idx, q_matrix)
#     return train_ds, valid_ds, test_ds



def get_datasets(
    mapping_path: str,
    q_matrix_path: str,
    item_count: int,
    concept_count: int,
    train_valid_csv: str,
    test_csv: str,
    fold: int = 0  # 当前使用的fold
):
    """
    一次性加载映射、Q矩阵，并返回交叉验证的 train/valid/test 数据集。
    """
    question2idx = load_question_mapping(mapping_path)
    q_matrix = load_q_matrix(q_matrix_path, None,
                           num_exercises=len(question2idx),
                           num_concepts=None)
    
    # 创建训练集和验证集
    train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, 
                         fold_mode='train', fold=fold)
    valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, 
                         fold_mode='valid', fold=fold)
    
    # 创建测试集
    test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)
    
    return train_ds, valid_ds, test_ds