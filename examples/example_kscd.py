import argparse
from wandb_train_test import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KSCD model')
    parser.add_argument('--model_name', type=str, default='kscd')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='assist2009')
    
    # Model hyperparameters
    parser.add_argument('--emb_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=1, help='Use wandb for tracking')
    # Save path
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Save directory, if None, use default path')

    args = parser.parse_args()
    
    params = vars(args)
    main(params)