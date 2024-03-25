import os
import time
import argparse

import torch
import torch.optim as optim

from model import SConvCNP
from modules import UNO
from utils import *
from data import *



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.model_seed)

    if args.data == 'sawtooth':
        sampler = CurveSampler(SawtoothKernel())
    elif args.data == 'periodic':
        sampler = CurveSampler(PeriodicKernel())
    elif args.data == 'matern52':
        sampler = CurveSampler(Matern52Kernel())
    elif args.data == 'rbf':
        sampler = CurveSampler(RBFKernel())
    elif args.data == 'square':
        sampler = CurveSampler(SquareWaveKernel())
    else:
        raise "Data Generator Not Implemented"

    model = SConvCNP(
        in_channels=1,
        out_channels=1,
        rho=UNO(),
        points_per_unit=64
        ).to(device)
    print_trainable_parameters(model)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
        )

    model, train_log_p = train(
        model=model,
        optimizer=optimizer,
        num_tasks=args.num_train_tasks,
        batch_size=args.train_batch_size,
        data_generator=sampler,
        device=device
        )
    print(f"Average log likelihood on training tasks: {train_log_p}")

    eval_log_p = eval(
        model=model,
        num_tasks=args.num_eval_tasks,
        batch_size=args.eval_batch_size,
        data_generator=sampler,
        device=device
        )

    print(f"Average log likelihood on unseen tasks: {eval_log_p}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=43)
    parser.add_argument('--data', default='sawtooth', choices=['sawtooth', 'periodic', 'matern52', 'rbf'])
    parser.add_argument('--train_batch_size', default=16)
    parser.add_argument('--eval_batch_size', default=16)
    parser.add_argument('--learning_rate', default=5e-4)
    parser.add_argument('--num_train_tasks', default=int(1e5))
    parser.add_argument('--num_eval_tasks', default=int(1e3))
    args = parser.parse_args()
    
    main(args)









