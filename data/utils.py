# PYCD/data/utils.py
import os
import json
import pandas as pd
import numpy as np
import ast

def convert_csv_to_json(csv_path, q_matrix, out_path):
    """
    将 CSV（包含 user_id, question_id, correct）转换为 JSON，
    并添加该题对应的 q_vector（来自 Q 矩阵）
    """
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        qid = int(row['question_id'])
        uid = int(row['user_id'])
        label = float(row['correct'])
        q_vec = q_matrix[qid].tolist()
        samples.append({
            'user_id': uid,
            'exer_id': qid,
            'correct': label,
            'q_vector': q_vec
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f)
    print(f"[Convert] {csv_path} → {out_path}")


def batch_convert_to_rcd_json(data_dir):
    """
    批量将 train_valid.csv 与 test.csv 转换为 JSON（附加 q_vector）
    要求存在：
    - id_mapping.json
    - q_matrix.csv
    """
    id_map_path = os.path.join(data_dir, 'id_mapping.json')
    qmat_path = os.path.join(data_dir, 'q_matrix.csv')
    train_valid_csv = os.path.join(data_dir, 'train_valid.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    train_json = os.path.join(data_dir, 'train_valid.json')
    test_json = os.path.join(data_dir, 'test.json')

    if all(os.path.exists(p) for p in [train_json, test_json]):
        print("[Skip] JSON 文件已存在")
        return

    with open(id_map_path, 'r', encoding='utf-8') as f:
        qid_map = json.load(f)['questions']  # 题目映射字典（如果需要）

    df = pd.read_csv(qmat_path)
    df['concept_ids'] = df['concept_ids'].apply(lambda x: ast.literal_eval(x))
    df = df.explode('concept_ids')
    max_qid = df['question_id'].max()
    max_kid = df['concept_ids'].max()

    q_matrix = np.zeros((max_qid + 1, max_kid + 1), dtype=np.float32)
    for _, row in df.iterrows():
        q_matrix[int(row['question_id']), int(row['concept_ids'])] = 1.0

    convert_csv_to_json(train_valid_csv, q_matrix, train_json)
    convert_csv_to_json(test_csv, q_matrix, test_json)
