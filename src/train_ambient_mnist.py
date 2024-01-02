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
import degradations
import utils
import ast
import argparse
import sys
import os
import time
import datetime

parser = argparse.ArgumentParser()
# Multiprocessing arguments
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

# training arguments
parser.add_argument("--num_kimgs", type=int, default=10000)
parser.add_argument("--num_kimgs_post", type=int, default=10000)
parser.add_argument("--num_kimgs_main", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr_post", type=float, default=1e-06)
parser.add_argument("--lr_main", type=float, default=1e-06)
# parser.add_argument("--lr_pre", type=float, default=1e-05)
parser.add_argument("--reg_parameter", type=float, default=1e-03)
parser.add_argument("--num_z", type=int, default=10)
parser.add_argument("--results_dir", type=str, default='')
parser.add_argument("--resume_from", type=str, default='')

# model arguments
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--main_model_type", type=str, default='ConvINN')
parser.add_argument("--main_model_args", type=ast.literal_eval, default={'num_conv_layers': [4,12], 'num_fc_layers': [4]})
parser.add_argument("--main_actnorm", type=lambda b:bool(int(b)), help="0 or 1")
parser.add_argument("--post_model_type", type=str, default='ConvINN')
parser.add_argument("--post_model_args", type=ast.literal_eval, default={'num_conv_layers': [4,12], 'num_fc_layers': [4]})
parser.add_argument("--post_actnorm", type=lambda b:bool(int(b)), help="0 or 1")

# data args
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})
parser.add_argument("--num_bits", type=int, default=0)


def train(gpu, args):

    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    def print0(*args, **kwargs):
        if gpu == 0:    print(*args, **kwargs)

    args.num_iters = args.num_kimgs * 1000 // args.batch_size
    args.num_iters_post = args.num_kimgs_post * 1000 // args.batch_size
    args.num_iters_main = args.num_kimgs_main * 1000 // args.batch_size

    args.data_args['input_shape'] = args.input_shape
    degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)
    train_dataset = getattr(dataset_tool, args.data_type)(train=True,  ambient=True, degradation=degradation, **args.data_args)
    test_dataset  = getattr(dataset_tool, args.data_type)(train=False, ambient=True, degradation=degradation, **args.data_args)
    test_dataset_clean  = getattr(dataset_tool, args.data_type)(train=False, **args.data_args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=4)

    main_model = getattr(models, args.main_model_type)(args.input_shape, **args.main_model_args)
    post_model = getattr(models, args.post_model_type)(args.input_shape, cond_shape=degradation.output_shape, **args.post_model_args)
    main_optimizer = torch.optim.Adam(main_model.trainable_parameters, lr=args.lr_main)
    post_optimizer = torch.optim.Adam(post_model.trainable_parameters, lr=args.lr_post)
    # tot_optimizer  = torch.optim.Adam(main_model.trainable_parameters + post_model.trainable_parameters, lr=args.lr)

    save_folder = utils.setup_saver(args.results_dir, main_model.identifier + f'-lr{args.lr_main}' + f'-{post_model.identifier}' + f'-lr{args.lr_post}' + f'-bits{args.num_bits}-{args.degradation_type}')
    print0(args.__dict__, file=open(f'{save_folder}/config.txt', 'w'))
    sys.stdout = utils.Logger(save_folder+'/log.txt')
    print0(args, flush=True)

    if gpu == 0:
        noisies = [train_dataset[i][0].detach().numpy() for i in range(16)]
        utils.save_images(np.stack([test_dataset_clean[i][0].detach().numpy() for i in range(16)]), f'{save_folder}/reals.png', imrange=[-0.5,0.5])
        utils.save_images(noisies, f'{save_folder}/noisies.png', imrange=[-0.5, 0.5])

    print("Pretraining posterior model")

    if gpu == 0:
        os.makedirs(f'{save_folder}/posterior-pretraining', exist_ok=True)
        y_tests = [test_dataset[i][0] for i in range(4)]
        start_time = time.time()
    idx = 0
    dloader = iter(train_dataloader)
    while True:
        try:    y = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            y = next(dloader)[0]

        # y = torch.cat([y]*3, dim=1)
        y = y.to(device)
        post_optimizer.zero_grad()
        loss,_ = post_model.get_loss(y, degradation, num_z=args.num_z, reg_parameter=args.reg_parameter, num_bits=args.num_bits)
        loss.backward()
        warmup_lr = args.lr_post * min(1, idx * args.batch_size / (10000 * 10))
        post_optimizer.param_groups[0]["lr"] = warmup_lr
        post_optimizer.step()

        if idx % 50 == 0:
            timesec = time.time() - start_time
            timesec = str(datetime.timedelta(seconds=int(timesec)))
            print0(f"kImg. : {idx*args.batch_size/1000:.2f}, time : {timesec} Curr. loss : {loss}")
        if idx % 500 == 0:
            xsamps = []
            if gpu == 0:
                for yt in y_tests:
                    # yt = torch.cat([yt]*3, dim=0)
                    yt = yt.reshape(1,*yt.shape).to(device)
                    x_samp = post_model.sample(4, yt, temp=1)
                    # x_samp = np.swapaxes(x_samp, 1,2)
                    # x_samp = np.swapaxes(x_samp, 2,3)
                    xsamps.append(x_samp)
                xsamps = np.concatenate(xsamps, axis=0)
                utils.save_images(xsamps, f'{save_folder}/posterior-pretraining/fakes_{(idx*args.batch_size//1000):06}.png', imrange=[-0.5,0.5])
                post_model.save(f'{save_folder}/posterior-pretraining/network_{(idx*args.batch_size//1000):06}.pt')

        idx += 1
        if idx >= args.num_iters_post:
            break

    print("Pretraining main model")
    if gpu == 0:
        os.makedirs(f'{save_folder}/main-pretraining', exist_ok=True)
        utils.save_images(np.stack([test_dataset_clean[i][0].detach().numpy() for i in range(16)]), f'{save_folder}/main-pretraining/reals.png', imrange=[-0.5,0.5])
        utils.save_images(noisies, f'{save_folder}/main-pretraining/noisies.png', imrange=[-0.5, 0.5])
    idx = 0
    dloader = iter(train_dataloader)
    while True:
        try:    x = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            x = next(dloader)[0]
        
        # x = torch.cat([x]*3, dim=1)
        x.to(device)
        if idx == 0 and args.main_actnorm:
            main_model.initialize_actnorm(x)
            main_model = main_model.to(device)
            idx += 1
            continue

        main_optimizer.zero_grad()
        loss = main_model.get_loss(x, num_bits=args.num_bits)
        loss.backward()
        warmup_lr = args.lr_main * min(1, idx * args.batch_size / (10000 * 10))
        main_optimizer.param_groups[0]["lr"] = warmup_lr
        main_optimizer.step()

        if idx % (50 // args.batch_size) == 0:
            timesec = time.time() - start_time
            timesec = str(datetime.timedelta(seconds=int(timesec)))
            print0(f"kImg. : {idx*args.batch_size/1000:.2f}, time : {timesec} Curr. loss : {loss}")
        if idx % (500 // args.batch_size) == 0:
            for temp in [0.5, 0.7, 1.0]:
                x_samp = main_model.sample(16, temp=temp)
                # x_samp = np.squeeze(x_samp)
                # x_samp = np.swapaxes(x_samp, 1,2)
                # x_samp = np.swapaxes(x_samp, 2,3)
                # x_samp = np.squeeze(x_samp)
                utils.save_images(x_samp, f'{save_folder}/main-pretraining/fakes_{(idx*args.batch_size//1000):06}_temp{temp}.png', imrange=[-0.5,0.5])
            main_model.save(f'{save_folder}/main-pretraining/network_{(idx*args.batch_size//1000):06}.pt')

        idx += 1
        if idx >= args.num_iters_main:
            break


    print("Jointly training main and posterior models")
    if gpu == 0:
        utils.save_images(np.stack([test_dataset_clean[i][0].detach().numpy() for i in range(16)]), f'{save_folder}/reals.png', imrange=[-0.5,0.5])
        utils.save_images(noisies, f'{save_folder}/noisies.png', imrange=[-0.5, 0.5])
    idx = 0
    dloader = iter(train_dataloader)
    while True:
        try:    y = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            y = next(dloader)[0]

        # y = torch.cat([y]*3, dim=1)
        y = y.to(device)
        # tot_optimizer.zero_grad()
        main_optimizer.zero_grad()
        post_optimizer.zero_grad()
        loss, x_posts = post_model.get_loss(y, degradation, num_z=args.num_z, reg_parameter=args.reg_parameter)
        for x_post in x_posts:
            loss += main_model.get_loss(x_post, num_bits=args.num_bits) / args.num_z
        loss.backward()
        # tot_optimizer.step()
        warmup_lr = args.lr_main * min(1, (idx + args.num_iters_main) * args.batch_size / (10000 * 10))
        main_optimizer.param_groups[0]["lr"] = warmup_lr
        warmup_lr = args.lr_post * min(1, (idx + args.num_iters_post) * args.batch_size / (10000 * 10))
        post_optimizer.param_groups[0]["lr"] = warmup_lr
        main_optimizer.step()
        post_optimizer.step()

        if idx % (5000 // args.batch_size) == 0:
            timesec = time.time() - start_time
            timesec = str(datetime.timedelta(seconds=int(timesec)))
            print0(f"kImg. : {idx*args.batch_size/1000:.2f}, time : {timesec} Curr. loss : {loss}")
        if idx % (50000 // args.batch_size) == 0:
            for temp in [0.5, 0.7, 1.0]:
                x_samp = main_model.sample(16, temp=temp)
                utils.save_images(x_samp, f'{save_folder}/fakes_{(idx*args.batch_size//1000):06}_temp{temp}.png', imrange=[-0.5,0.5])
            xsamps = []
            if gpu == 0:
                for yt in y_tests:
                    yt = yt.reshape(1,*yt.shape).to(device)
                    x_samp = post_model.sample(4, yt, temp=1)
                    xsamps.append(x_samp)
                xsamps = np.concatenate(xsamps, axis=0)
                utils.save_images(xsamps, f'{save_folder}/fakes_{(idx*args.batch_size//1000):06}_postsamp.png', imrange=[-0.5,0.5])
            main_model.save(f'{save_folder}/main-network_{(idx*args.batch_size//1000):06}.pt')
            post_model.save(f'{save_folder}/post-network_{(idx*args.batch_size//1000):06}.pt')

        idx += 1
        if idx >= args.num_iters:
            break


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

    train(0, args)