import argparse
from wandb_train_test import main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='训练SCD模型')
    parser.add_argument('--model_name', type=str, default='scd')
    parser.add_argument('--dataset', type=str, default='assist2009')
    # 模型超参数
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--parameter_epochs', type=int, default=50, help='Parameter Epoch')
    parser.add_argument('--interaction_epochs', type=int, default=5, help='Interaction Epoch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)

    parser.add_argument('--population_size', type=int, default=20, help='Group size')
    parser.add_argument('--ngen', type=int, default=5, help='The number of genetic algorithm iterations')
    parser.add_argument('--cxpb', type=float, default=0.5, help='Crossover probability')
    parser.add_argument('--mutpb', type=float, default=0.1, help='Probability of variation')

    parser.add_argument('--use_wandb', type=int, default=1, help='是否使用wandb')
    # 保存路径
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录，如果为None则使用 examples/model_save/neuralcdm')
    args = parser.parse_args()
    
    params=vars(args)
    main(params)
