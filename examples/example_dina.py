import argparse
from wandb_train_test import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DINA model')
    parser.add_argument('--model_name', type=str, default='dina')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='assist2009')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=40, 
                        help='Hidden dimension (knowledge state dimension)')
    parser.add_argument('--concept_dim', type=int, default=None, 
                        help='Concept dimension (defaults to number of concepts in the dataset)')
    parser.add_argument('--ste', type=int, default=1, 
                        help='Whether to use Straight-Through Estimator (1 for true, 0 for false)')
    parser.add_argument('--max_slip', type=float, default=0.4, 
                        help='Maximum slip parameter')
    parser.add_argument('--max_guess', type=float, default=0.4, 
                        help='Maximum guess parameter')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)
    
    # WandB settings
    parser.add_argument('--use_wandb', type=int, default=0, help='Use wandb for tracking')
    
    # Save path
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Save directory, if None, use default path')

    args = parser.parse_args()
    
    params = vars(args)
    main(params)