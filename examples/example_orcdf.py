import argparse
import torch
from wandb_train_test import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ORCDF（kancd） model')
    parser.add_argument('--model_name', type=str, default='orcdf',
                        help='ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems')
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='frcsub')

    # 模型超参数
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dims1', type=int, default=512)
    parser.add_argument('--hidden_dims2', type=int, default=256)
    parser.add_argument('--ssl_temp', type=float, default=0.5)
    parser.add_argument('--ssl_weight', type=float, default=1e-3)
    parser.add_argument('--flip_ratio', type=float, default=0.15)
    parser.add_argument('--gcn_layers', type=int, default=3, help='numbers of gcn layers')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='edge drop probability')
    parser.add_argument('--if_type', type=str, default='kancd',help='only use kancd')
    parser.add_argument('--mode', type=str, default='all', help='use for Ablation Study')
    parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')


    # 通用训练参数
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_wandb', type=int, default=0)
    # Save path
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory, if None, use default path')

    args = parser.parse_args()
    params = vars(args)

    main(params)