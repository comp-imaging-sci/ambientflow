""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.optim
import numpy as np
import dataset_tool
import models
import utils
import ast
import os
import argparse
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import datetime

import torch.nn.parallel as parallel

parser = argparse.ArgumentParser()

# multiprocessing arguments
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

# model arguments
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--model_type", type=str, default='ConvINN')
parser.add_argument("--model_args", type=ast.literal_eval, default={'num_conv_layers': [4,12], 'num_fc_layers': [4]})
parser.add_argument("--actnorm", type=lambda b:bool(int(b)), help="0 or 1")

# training arguments
parser.add_argument("--num_kimgs", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-05)
parser.add_argument("--results_dir", type=str, default='')
parser.add_argument("--resume_from", type=str, default='')

# data args
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--num_bits", type=int, default=0)

def train(gpu, args):

    def print0(*args, **kwargs):
        if gpu == 0:    print(*args, **kwargs)

    args.num_iters = args.num_kimgs * 1000 // args.batch_size
    
    # initialize process
    # rank = args.nr * args.gpus + gpu	                          
    # dist.init_process_group(                                   
    # 	backend='nccl',                                         
   	# 	init_method='env://',                                   
    # 	world_size=args.world_size,                              
    # 	rank=rank                                               
    # ) 

    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    args.data_args['input_shape'] = args.input_shape
    train_dataset = getattr(dataset_tool, args.data_type)(train=True, **args.data_args)
    test_dataset  = getattr(dataset_tool, args.data_type)(train=False, **args.data_args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = getattr(models, args.model_type)(args.input_shape, **args.model_args, device=device).to(device)
    optimizer = torch.optim.Adamax(model.trainable_parameters, lr=args.lr)

    # Configure saving
    save_folder = utils.setup_saver(args.results_dir, model.identifier+f'-lr{args.lr}-bits{args.num_bits}')
    print0(args.__dict__, file=open(f'{save_folder}/config.txt', 'w'))
    sys.stdout = utils.Logger(save_folder+'/log.txt')
    print0(args, flush=True)

    if gpu == 0:
        utils.save_images(np.stack([train_dataset[i][0].detach().numpy() for i in range(16)], axis=0), f'{save_folder}/reals.png', imrange=[-0.5,0.5])

    if args.resume_from != '':
        print0(f"Resuming training from {args.resume_from} ...")
        model.load(args.resume_from)
        model.to(device)

    print0("Training vanilla model")
    idx = 0
    dloader = iter(train_dataloader)
    start_time = time.time()
    while True:
        try:    x = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            x = next(dloader)[0]
        
        # x = torch.cat([x]*3, dim=1)
        x.to(device, non_blocking=True)
        if args.resume_from == '':
            if idx == 0 and args.actnorm:
                model.initialize_actnorm(x)
                model = model.to(device)
                # model = parallel.DistributedDataParallel(model, device_ids=[gpu])
                idx += 1
                continue

        optimizer.zero_grad()
        loss = model.get_loss(x, num_bits=args.num_bits)
        loss.backward()
        # warmup_lr = args.lr * min(1, idx * args.batch_size / (10000 * 10))
        warmup_lr = args.lr * min(1, idx * args.batch_size / (50000 * 10))
        optimizer.param_groups[0]["lr"] = warmup_lr
        optimizer.step()
        print_freq = int(np.ceil(5000 / args.batch_size))
        save_freq  = int(np.ceil(100000 / args.batch_size))

        if (idx < print_freq and idx % (print_freq // 5) == 0) or (idx % print_freq == 0):
            timesec = time.time() - start_time
            timesec = str(datetime.timedelta(seconds=int(timesec)))
            print0(f"kImg. : {idx*args.batch_size/1000:.2f}, time : {timesec} Curr. loss : {loss}")
        if (idx < save_freq and idx % (save_freq // 5) == 0) or (idx % save_freq == 0):
            for temp in [0.5, 0.7, 1.0]:
                x_samp = model.sample(16, temp=temp)
                # x_samp = np.squeeze(x_samp)
                # x_samp = np.swapaxes(x_samp, 1,2)
                # x_samp = np.swapaxes(x_samp, 2,3)
                # x_samp = np.squeeze(x_samp)
                if gpu == 0:
                    utils.save_images(x_samp, f'{save_folder}/fakes_{(idx*args.batch_size//1000):06}_temp{temp}.png', imrange=[-0.5,0.5])
            model.save(f'{save_folder}/network_{(idx*args.batch_size/1000):06}.pt')
            opt_state = {
                'idx': idx,
                'opt': optimizer.state_dict(),
                'lr' : optimizer.param_groups[0]["lr"]
            }
            torch.save(opt_state, f'{save_folder}/optimizer_{(idx*args.batch_size//1000):06}.pt')

        idx += 1
        if idx >= args.num_iters:
            break


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

    train(0, args)

    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # mp.spawn(train, nprocs=args.gpus, args=(args,))


