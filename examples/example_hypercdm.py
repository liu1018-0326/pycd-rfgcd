import argparse
from wandb_train_test import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练 HyperCDM 模型')
    parser.add_argument('--model_name', type=str, default='hypercdm')
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='assist2009')

    # HyperCDM 专属超参数
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='映射到超图传播的特征维度')
    parser.add_argument('--emb_dim', type=int, default=16,
                        help='初始学生/题目/知识点嵌入维度')
    parser.add_argument('--layers', type=int, default=5,
                        help='超图卷积网络层数')

    # 通用训练超参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备，可选 cpu 或 cuda')

    parser.add_argument('--use_wandb', type=int, default=0,
                        help='是否使用 wandb 记录日志')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存目录，若为 None 则使用 examples/model_save/hypercdm')

    args = parser.parse_args()
    params = vars(args)
    main(params)