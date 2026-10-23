import torch
import argparse
from wandb_train_test import main


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='训练ICDM模型')
    parser.add_argument('--model_name', type=str, default='icdm')
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='frcsub')
    
    # 模型超参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--method', default='icdm-glif', type=str, help='方法类型')
    parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
    parser.add_argument('--gcnlayers', type=int, default=2, help='gcn层数')
    parser.add_argument('--dim', type=int, default=64,help='dimension of hidden layer')
    parser.add_argument('--exp_type', type=str, default='cdm', help='模型类型')
    parser.add_argument('--agg_type', type=str, default='mean', help='the type of aggregator')
    parser.add_argument('--cdm_type', type=str, default='glif', help='the inherent CDM')
    parser.add_argument('--khop', default=2, type=int)
    parser.add_argument('--weight_reg', type=float, default=0.001, help='the inherent CDM')
    
    # 使用wandb
    parser.add_argument('--use_wandb', type=int, default=0, help='Use wandb for tracking')
    # 保存路径
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录，如果为None则使用 examples/model_save/neuralcdm')

    args = parser.parse_args()
    params=vars(args)
    #print(parmas)
    main(params)
