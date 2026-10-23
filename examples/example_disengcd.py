import argparse
import torch
from wandb_train_test import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练DisenGCD模型')
    parser.add_argument('--model_name', type=str, default='disengcd')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='assist2012')
    
    # 模型超参数
    parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
    parser.add_argument('--dropout1', type=float, default=0.5)
    parser.add_argument('--dropout2', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--k', type=float, default=1, help='sampling')
    parser.add_argument('--lam_seq', type=float, default=0.9, help='threshold')
    parser.add_argument('--lam_res', type=float, default=0.9, help='threshold')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--alr', type=float, default=0.0003, help='learning rate for architecture parameters')
    parser.add_argument('--use_wandb', type=int, default=0, help='是否使用wandb')
    parser.add_argument('--save_dir', type=str, default=None)
    
    args = parser.parse_args()
    params = vars(args)
    
    main(params)