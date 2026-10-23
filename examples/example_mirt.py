import argparse
from wandb_train_test import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练MIRT模型')
    parser.add_argument('--model_name', type=str, default='mirt')
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='assist2009')
    
    # 模型超参数
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--a_range', type=int, default=None, help='区分度参数范围约束，默认为None')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=0, help='使用wandb进行训练')
    # 保存路径
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录，如果为None则使用默认路径')

    args = parser.parse_args()
    
    params=vars(args)
    main(params)
