import torch
import argparse
from wandb_train_test import main


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train RFGCD model')
    parser.add_argument('--model_name', type=str, default='rfgcd')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='frcsub')

    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--method', default='icdm-glif', type=str, help='method type')
    
    parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
    parser.add_argument('--gcnlayers', type=int, default=2, help='number of GCN layers')
    parser.add_argument('--dim', type=int, default=64,help='dimension of hidden layer')
    parser.add_argument('--exp_type', type=str, default='cdm', help='model type')
    parser.add_argument('--agg_type', type=str, default='mean', help='the type of aggregator')
    parser.add_argument('--cdm_type', type=str, default='glif', help='the inherent CDM')
    parser.add_argument('--khop', default=2, type=int)
    parser.add_argument('--weight_reg', type=float, default=0.001, help='the inherent CDM')
    
    # wandb logging
    parser.add_argument('--use_wandb', type=int, default=0, help='Use wandb for tracking')
    # Save path
    parser.add_argument('--save_dir', type=str, default=None, help='save directory; defaults to examples/model_save/neuralcdm')

    args = parser.parse_args()
    # Convert dtype string to torch.dtype if passed from command line
    if isinstance(args.dtype, str):
        dtype_map = {
            'torch.float32': torch.float32, 'torch.float': torch.float32,
            'torch.float64': torch.float64, 'torch.double': torch.float64,
        }
        args.dtype = dtype_map.get(args.dtype, torch.float64)
    params=vars(args)
    #print(parmas)
    main(params)
