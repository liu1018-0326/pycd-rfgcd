import argparse
from wandb_train_test import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练KaNCD模型')
    parser.add_argument('--model_name', type=str, default='kancd')
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='assist2009')
    
    # 模型超参数
    parser.add_argument('--hidden_dims1', type=int, default=512)
    parser.add_argument('--hidden_dims2', type=int, default=256)
    parser.add_argument('--dropout1', type=float, default=0.5)
    parser.add_argument('--dropout2', type=float, default=0.5)
    parser.add_argument('--emb_dim', type=int, default=20)
    parser.add_argument('--mf_type', type=str, default='ncf2')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=1, help='使用wandb进行训练')
    # 保存路径
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录，如果为None则使用 examples/model_save/kancd')
    
    args = parser.parse_args()
    
    params=vars(args)
    main(params)
