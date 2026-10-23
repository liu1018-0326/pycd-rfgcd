import argparse
from wandb_train_test import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IRT model')
    parser.add_argument('--model_name', type=str, default='irt')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='assist2009')
    
    # Model hyperparameters
    parser.add_argument('--value_range', type=float, default=None, 
                        help='Constraint range for theta and b parameters, default is None')
    parser.add_argument('--a_range', type=float, default=None, 
                        help='Constraint range for discrimination parameter, default is None')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    # Wandb settings
    parser.add_argument('--use_wandb', type=int, default=0, help='Use wandb for tracking')
    # Save path
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Save directory, if None, use default path')

    args = parser.parse_args()
    
    params = vars(args)
    main(params)