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
import models2
import degradations
import utils
import ast
import argparse
import sys
import os
import time
import datetime
# import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
# import wandb
import socket
import sparsifiers
# from torch.profiler import profile, record_function, ProfilerActivity

class AttrDataParallel(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
#     def __setatptr__(self, name, value):
#         try:
#             return super().__setattr__(name, value)
#         except AttributeError:
#             return setattr(self.module, name, value)

parser = argparse.ArgumentParser()
# Multiprocessing arguments
parser.add_argument('--rank', default=0, type=int, help='Current rank')
parser.add_argument('-n', '--num_nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--num_gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('--port', type=str, default='12355')

# training arguments
parser.add_argument("--num_kimgs", type=float, default=10000)
parser.add_argument("--num_kimgs_post", type=float, default=10000)
parser.add_argument("--num_kimgs_main", type=float, default=10000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr_post", type=float, default=1e-06)
# parser.add_argument("--lr_main", type=float, default=1e-06)
parser.add_argument("--lr", type=float, default=1e-06)
# parser.add_argument("--lr_pre", type=float, default=1e-05)
parser.add_argument("--reg_parameter", type=float, default=1e-03)
parser.add_argument("--num_z", type=int, default=10)
parser.add_argument("--results_dir", type=str, default='')
parser.add_argument("--resume_from", type=str, default='')
parser.add_argument("--tiled_for_loss", type=lambda b:bool(int(b)), default=0, help="Whether to tile the data batch while computing loss or loop over various zs")
parser.add_argument("--importance_weighting", type=int, default=0, help="Whether or not to do importance weighting for the gradient update")
parser.add_argument("--schedule_sparsity_weight", type=lambda b:bool(int(b)), help="0 or 1")
parser.add_argument("--benchmark_cudnn", type=lambda b:bool(int(b)), help="0 or 1", default=0)

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

# sparsity regularization related args
parser.add_argument("--cutoff_dim", type=float, default=0)
parser.add_argument("--sparsifier_type", type=str, default='WaveletTransform')
parser.add_argument("--sparsifier_args", type=ast.literal_eval, default={'level': 2, 'wavelet': 'haar'})
parser.add_argument("--sparsity_weight", type=float, default=0.0, help="")
parser.add_argument("--threshold_weight", type=float, default=0.0, help="")

def train(gpu, args):

    # args.lr_post = args.lr_main = args.lr
    args.lr_main = args.lr
    # args.lr_post = args.lr * 100

    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    )
    # print("Im here")
    # Configure wandb
    if rank == 0:
        server_name = socket.gethostname().split('.')[0]
        # wandb.init(mode="disabled", settings=wandb.Settings(start_method="fork"),
        #     project=f'{server_name}-{args.num_gpus}gpu-{args.data_type}-deg{args.degradation_type}-{args.main_model_type}-{args.post_model_type}-reg{args.reg_parameter}-bit{args.num_bits}-lr{args.lr}-nz{args.num_z}-impw{args.importance_weighting}', entity='kvarun95')
        # wandb.config = {
        #     "num_gpus"      : args.num_gpus,
        #     "main_lr"       : args.lr_main,
        #     "post_lr"       : args.lr_post,
        #     "kimgs"         : int(args.num_kimgs),
        #     "reg"           : args.reg_parameter,
        #     "bits"          : args.num_bits,
        #     "num_z"         : args.num_z,
        #     "batch_size"    : args.eff_batch_size,
        #     "deg"           : args.degradation_type,
        #     "deg_args"      : args.degradation_args,
        #     "input_shape"   : args.input_shape,
        # }
    # print("Im here 2")

    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    # print("Im here 3")

    def print0(*args, **kwargs):
        if rank == 0:    print(*args, **kwargs)

    args.num_iters = int( args.num_kimgs * 1000 / args.eff_batch_size )
    args.num_iters_post = int( args.num_kimgs_post * 1000 / args.eff_batch_size ) if args.num_kimgs_post else 1
    args.num_iters_main = int( args.num_kimgs_main * 1000 / args.eff_batch_size ) if args.num_kimgs_main else 1

    # forward model (degradation)
    args.data_args['input_shape'] = args.input_shape
    degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits, device=device)
    # print("Im here 4")

    # sparsifier
    sp_trans = getattr(sparsifiers, args.sparsifier_type)(args.input_shape, **args.sparsifier_args)
    args.post_model_args['sparsifier'] = sp_trans
    args.post_model_args['sparsity_weight'] = args.sparsity_weight

    # main and posterior models
    main_model = getattr(models2, args.main_model_type)(args.input_shape, **args.main_model_args, device=device)
    # print("Im here 5")
    post_model = getattr(models2, args.post_model_type)(args.input_shape, cond_shape=degradation.output_shape, **args.post_model_args, device=device)
    # print("Im here 6")

    # Resume training from previous iteration
    if args.resume_from != '':
        print0(f"Resuming training from {args.resume_from} ...")
        main_model.load(args.resume_from)
        post_resume_from = os.path.split(args.resume_from)
        post_resume_from = os.path.join(post_resume_from[0], 'post-'+post_resume_from[1][5:])
        post_model.load(post_resume_from)
    main_model.to(device)
    post_model.to(device)
    # print("Im here 7")

    # optimizer
    main_optimizer = torch.optim.Adam(main_model.trainable_parameters, lr=args.lr_main)
    post_optimizer = torch.optim.Adam(list(post_model.trainable_parameters), lr=args.lr_post)
    # tot_optimizer  = torch.optim.Adam(main_model.trainable_parameters + post_model.trainable_parameters, lr=args.lr)
    # print("Im here 9")

    # wrap models with DDP
    main_model = AttrDataParallel(main_model, device_ids=[gpu])
    post_model = AttrDataParallel(post_model, device_ids=[gpu])
    # print("Im hre 10")

    # data loaders using the distributed sampler
    train_dataset = getattr(dataset_tool, args.data_type)(train=True,  ambient=True, degradation=degradation, **args.data_args)
    test_dataset  = getattr(dataset_tool, args.data_type)(train=False, ambient=True, degradation=degradation, **args.data_args)
    test_dataset_clean  = getattr(dataset_tool, args.data_type)(train=False, **args.data_args)
    # print("Im here 11")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    # print("im here 12")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size      = args.batch_size, 
                            shuffle         = False, 
                            num_workers     = 4*args.world_size,
                            pin_memory      = True,
                            sampler         = train_sampler)
    # test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # print("Im here 13")

    if rank == 0:
        save_folder = utils.setup_saver(args.results_dir, f'{server_name}-{args.num_gpus}gpu-deg{args.degradation_type}-{args.main_model_type}-{args.post_model_type}-reg{args.reg_parameter}-spparam{args.sparsity_weight}-thw{args.threshold_weight}-bit{args.num_bits}-lr{args.lr}-nz{args.num_z}-impw{args.importance_weighting}')
        # print("Im here 14")
        print0(args.__dict__, file=open(f'{save_folder}/config.txt', 'w'))
        sys.stdout = utils.Logger(save_folder+'/log.txt')
        print0(args)
        print(save_folder)

        noisies = [degradation.rev(train_dataset[i][0].to(device), mode='real').cpu().detach().numpy() for i in range(16)]
        _ = utils.save_images(np.stack([test_dataset_clean[i][0].detach().numpy() for i in range(16)]), f'{save_folder}/reals.png', imrange=[-0.5,0.5])
        _ = utils.save_images(noisies, f'{save_folder}/noisies.png', imrange=[-0.5-degradation.std*np.sqrt(2/np.pi), 0.5])

    # define importance weighting functions
    # def imp_weighting1(loss):
    #     loss -= np.log(args.num_z)
    #     loss = torch.logsumexp(loss, 0)
    #     loss = torch.mean(loss)
    #     loss_weight = 1
    #     return loss, loss_weight

    def imp_weighting1(loss):
        importance_weights = torch.softmax(loss, dim=0).detach()
        loss = loss * importance_weights
        loss = torch.mean(torch.sum(loss, dim=0))
        loss_weight = 1
        return loss, loss_weight

    def imp_weighting2(loss):
        loss -= np.log(args.num_z*args.batch_size)
        loss = torch.logsumexp(loss, [0,1])
        total_loss = torch.zeros_like(loss)
        dist.barrier()

        # losses_to_scatter = [loss.clone().detach() for _ in range(args.world_size)]
        # gathered_losses = list(torch.empty([4], dtype=loss.dtype).to(device).chunk(4))
        # dist.all_to_all(gathered_losses, losses_to_scatter)
        # dist.barrier()
        # total_l = torch.logsumexp(torch.Tensor(gathered_losses), 0).detach()
        # log_weight = loss.detach() - total_loss.detach()
        # loss_weight = torch.exp(log_weight)
        # return loss, loss_weight

        if rank == 0:
            gathered_losses = [torch.zeros_like(loss) for _ in range(args.world_size)]
            dist.gather(loss, gathered_losses, dst=0)
            total_l = torch.logsumexp(torch.Tensor(gathered_losses), 0).detach()
            dist.barrier()
            # scattered_losses = [total_l.clone() for _ in range(args.world_size)]
            # dist.scatter(total_loss, scattered_losses, src=0)
            # dist.broadcast(total_l, src=0)
            dist.reduce_scatter(total_loss, [total_l])
            dist.barrier()
        else:
            dist.gather(loss, dst=0)
            dist.barrier()
            # dist.broadcast(total_l, src=0)
            dist.barrier()

        log_weight = loss.detach() - total_loss.detach()
        loss_weight = torch.exp(log_weight)
        return loss, loss_weight

    # Use in the case of single GPU
    def imp_weighting2_single(loss):
        importance_weights = torch.softmax(loss.flatten(), dim=0).detach().reshape(loss.shape)
        loss *= importance_weights
        loss = torch.mean(loss)
        loss_weight = 1
        return loss, loss_weight

    # Switch on CUDNN benchmarking
    if args.benchmark_cudnn:
        torch.backends.cudnn.benchmark = True

    ################################################################
    print0("Pretraining posterior model")
    if rank == 0:
        os.makedirs(f'{save_folder}/posterior-pretraining', exist_ok=True)
        y_tests = [test_dataset[i][0] for i in range(4)]
        y_tests = [yt.reshape(1,*yt.shape).to(device, non_blocking=True) for yt in y_tests]
        y_tests = [degradation.rev(yt) for yt in y_tests]
        start_time = time.time()
        # print("im here 15")

    print_freq = int(np.ceil(2500 / args.eff_batch_size))
    save_freq  = int(np.ceil(50000 / args.eff_batch_size))
    idx = 0
    dloader = iter(train_dataloader)
    while args.num_iters_post and (args.resume_from == ''):
        try:    y = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            y = next(dloader)[0]
        # print("Im here 16")

        x = degradation.rev(y.to(device), use_device=True).detach()
        # print("Im here 17")
        if idx == 0 and args.post_actnorm:
            # x_in = degradation.rev(y.to(device), mode='real', use_device=True).detach()
            # post_model.initialize_actnorm(x_in, x)
            post_model.initialize_actnorm(y.to(device), degradation)
            idx += 1
            continue

        # y = torch.cat([y]*3, dim=1)
        y = y.to(device, non_blocking=True)
        post_optimizer.zero_grad()
        # print("Im here 17")
        loss,_ = post_model.get_loss(y, degradation, num_z=args.num_z, reg_parameter=1., num_bits=args.num_bits, tiled=args.tiled_for_loss, cutoff_dim=args.cutoff_dim)
        # print("Im here 18")

        # importance weighting
        # if args.importance_weighting == 1:
        #     loss, loss_weight = imp_weighting1(loss)
        # elif args.importance_weighting == 2:
        #     loss, loss_weight = imp_weighting2(loss)
        # else: loss_weight = 1

        loss.backward()
        warmup_lr = args.lr_post * min(1, idx * args.eff_batch_size / (10000 * 10))
        post_optimizer.param_groups[0]["lr"] = warmup_lr
        post_optimizer.step()
        if rank == 0:
            if (idx < print_freq and idx % max(1, print_freq // 5) == 0) or (idx % print_freq == 0):
                kimg = idx*args.eff_batch_size/1000
                timesec = time.time() - start_time
                timesec = str(datetime.timedelta(seconds=int(timesec)))
                print0(f"kImg. : {kimg:.2f}, time : {timesec} Curr. loss : {loss}")
                # wandb.log({"post_loss": loss}, step=int(kimg))
            if (idx < save_freq and idx % max(1, save_freq // 5) == 0) or (idx % save_freq == 0):
                kimg = idx*args.eff_batch_size // 1000
                xsamps = []
                for yt in y_tests:
                    # # yt = torch.cat([yt]*3, dim=0)
                    # yt = yt.reshape(1,*yt.shape).to(device, non_blocking=True)
                    # yt = degradation.rev(yt)
                    # # yt = torch.cat([yt]*4, dim=0)
                    x_samp = post_model.sample(4, yt, temp=1)
                    # x_samp = np.swapaxes(x_samp, 1,2)
                    # x_samp = np.swapaxes(x_samp, 2,3)
                    xsamps.append(x_samp)
                xsamps = np.concatenate(xsamps, axis=0)
                _ = utils.save_images(xsamps, f'{save_folder}/posterior-pretraining/fakes_{kimg:06}.png', imrange=[-0.5,0.5])
                # wandb.log({"post_example": wandb.Image(f'{save_folder}/posterior-pretraining/fakes_{kimg:06}.png')}, step=kimg)
                post_model.save(f'{save_folder}/posterior-pretraining/network_{kimg:06}.pt')
                
        idx += 1
        if idx >= args.num_iters_post:
            break

    print0("Pretraining main model")
    torch.cuda.empty_cache()
    if rank == 0:
        os.makedirs(f'{save_folder}/main-pretraining', exist_ok=True)
        _ = utils.save_images(np.stack([test_dataset_clean[i][0].detach().numpy() for i in range(16)]), f'{save_folder}/main-pretraining/reals.png', imrange=[-0.5,0.5])
        _ = utils.save_images(noisies, f'{save_folder}/main-pretraining/noisies.png', imrange=[-0.5, 0.5])
    idx = 0
    dloader = iter(train_dataloader)
    while args.num_iters_main and (args.resume_from == ''):
        try:    x = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            x = next(dloader)[0]

        # x = torch.cat([x]*3, dim=1)
        x = degradation.rev(x.to(device), mode='real', use_device=True).detach()
        if idx == 0 and args.main_actnorm:
            main_model.initialize_actnorm(x)
            # main_model = main_model.to(device)
            idx += 1
            continue

        main_optimizer.zero_grad()
        loss = main_model.get_loss(x, num_bits=args.num_bits)

        # if args.importance_weighting == 1:
        #     loss, loss_weight = imp_weighting1(loss)
        # elif args.importance_weighting == 2:
        #     loss, loss_weight = imp_weighting2(loss)
        # else: loss_weight = 1

        loss.backward()
        warmup_lr = args.lr_main * min(1, idx * args.eff_batch_size / (10000 * 10))
        main_optimizer.param_groups[0]["lr"] = warmup_lr
        main_optimizer.step()

        if idx % int(np.ceil(args.num_iters_main / 50)) == 0 and rank == 0:
            kimg = idx*args.eff_batch_size / 1000
            timesec = time.time() - start_time
            timesec = str(datetime.timedelta(seconds=int(timesec)))
            print0(f"kImg. : {kimg:.2f}, time : {timesec} Curr. loss : {loss}")
            # wandb.log({"main_loss": loss}, step=int(kimg))
        if idx % int(np.ceil(args.num_iters_main / 10)) == 0 and rank == 0:
            kimg = idx*args.eff_batch_size // 1000
            for temp in [0.5, 0.7, 1.0]:
                x_samp = np.concatenate([main_model.sample(4, temp=temp) for _ in range(4)])
                # x_samp = np.squeeze(x_samp)
                # x_samp = np.swapaxes(x_samp, 1,2)
                # x_samp = np.swapaxes(x_samp, 2,3)
                # x_samp = np.squeeze(x_samp)
                _ = utils.save_images(x_samp, f'{save_folder}/main-pretraining/fakes_{kimg:06}_temp{temp}.png', imrange=[-0.5,0.5])
                # wandb.log({f"main_example_{temp}": wandb.Image(f'{save_folder}/main-pretraining/fakes_{kimg:06}_temp{temp}.png')}, step=kimg)
            main_model.save(f'{save_folder}/main-pretraining/network_{kimg:06}.pt')

        idx += 1
        if idx >= args.num_iters_main:
            break

    print0("Jointly training main and posterior models")
    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()
    # if rank == 0:
    #     _ = utils.save_images(np.stack([test_dataset_clean[i][0].detach().numpy() for i in range(16)]), f'{save_folder}/reals.png', imrange=[-0.5,0.5])
    #     _ = utils.save_images(noisies, f'{save_folder}/noisies.png', imrange=[-0.5, 0.5])
    idx = 0
    dloader = iter(train_dataloader)
    print_freq = int(np.ceil(10000 / args.eff_batch_size))
    save_freq  = int(np.ceil(100000 / args.eff_batch_size))
    while args.num_iters:
        # t = time.time()
        try:    y,labels = next(dloader)
        except StopIteration: 
            dloader = iter(train_dataloader)
            y,labels = next(dloader)

        # torch.cuda.synchronize()
        # print("Dataloader  get next :", time.time() - t)
        # t = time.time()

        # sparsity regularization scheduling
        if args.schedule_sparsity_weight:
            kimg = idx*args.eff_batch_size // 1000
            # sparsity_weight = (kimg < 1000)*0 + (kimg >= 1000)*args.sparsity_weight*(1-np.exp(-abs(kimg-1000)/1000))
            sparsity_weight = args.sparsity_weight*(1-np.exp(-kimg/100))
            # post_model.change_sparsity_weight(sparsity_weight)
        else: sparsity_weight = args.sparsity_weight
        # torch.cuda.synchronize()
        # print("Sparsity schedule :", time.time() - t)
        # t = time.time()

        # y = torch.cat([y]*3, dim=1)
        y = y.to(device, non_blocking=True)
        # torch.cuda.synchronize()
        # print("Data to device", time.time() - t)
        # t = time.time()
        # tot_optimizer.zero_grad()
        main_optimizer.zero_grad(set_to_none=True)
        post_optimizer.zero_grad(set_to_none=True)
        # torch.cuda.synchronize()
        # print("Zero grad :", time.time() - t)
        # t = time.time()
        with torch.cuda.amp.autocast():
            post_loss, x_posts = post_model.get_loss(y, degradation, num_z=args.num_z, reg_parameter=args.reg_parameter, num_bits=args.num_bits, tiled=args.tiled_for_loss, importance_weighting=args.importance_weighting)
            # torch.cuda.synchronize()
            # print("Post loss :", time.time() - t)
            # t = time.time()
            x_posts,labels = train_dataset.deaugment(x_posts, labels)
            main_loss = main_model.get_loss(x_posts, num_bits=args.num_bits, importance_weighting=args.importance_weighting)
            # torch.cuda.synchronize()
            # print("Main loss :", time.time() - t)
            # t = time.time()
            loss = main_loss + post_loss

            # Importance weighting
            if args.importance_weighting == 1:
                loss, loss_weight = imp_weighting1(loss)
            elif args.importance_weighting == 2 and args.num_gpus == 1:
                loss, loss_weight = imp_weighting2_single(loss)
            elif args.importance_weighting == 2 and args.num_gpus != 1:
                loss, loss_weight = imp_weighting2(loss)
            else: loss_weight = 1
            # torch.cuda.synchronize()
            # print("Imp weights : ", time.time() - t)
            # t = time.time()


            # Sparsifying penalty
            # sparsifying_penalty = utils.total_variation(x_posts)
            # sparsifying_penalty = 0
            # coeffs = sp_trans(x_posts.view(-1, *args.input_shape)+0.5)
            if args.sparsity_weight:
                sparsifying_penalty = sparsifiers.sparsifying_regularizer2(x_posts.view(-1, *args.input_shape)+0.5, sp_trans, threshold_weight=args.threshold_weight)
            else: sparsifying_penalty = 0
            # torch.cuda.synchronize()
            # print("sparsifying_penalty :", time.time() - t)
            # t = time.time()
            loss_full = loss * loss_weight + sparsity_weight* sparsifying_penalty

        scaler.scale(loss_full).backward()
        # torch.cuda.synchronize()
        # print("Backward : ", time.time() - t)
        # t = time.time()
        # tot_optimizer.step()
        warmup_lr = args.lr_main * min(1, (idx + args.num_iters_main) * args.eff_batch_size / (10000 * 10))
        main_optimizer.param_groups[0]["lr"] = warmup_lr
        warmup_lr = args.lr_post * min(1, (idx + args.num_iters_post) * args.eff_batch_size / (10000 * 10))
        post_optimizer.param_groups[0]["lr"] = warmup_lr
        # main_optimizer.step()
        scaler.step(main_optimizer)
        # post_optimizer.step()
        scaler.step(post_optimizer)
        # torch.cuda.synchronize()
        # print("Optimizer step :", time.time() - t)
        # t = time.time()
        scaler.update()

        # Logging
        if rank == 0:    
            if (idx < print_freq and idx % (print_freq // 10) == 0) or (idx % print_freq == 0):
                kimg = idx*args.eff_batch_size/1000
                timesec = time.time() - start_time
                timesec = str(datetime.timedelta(seconds=int(timesec)))
                if args.importance_weighting == 1:
                    post_loss_summ = torch.mean(torch.logsumexp(post_loss, 0))
                    main_loss_summ = torch.mean(torch.logsumexp(main_loss, 0))
                elif args.importance_weighting == 2:
                    post_loss_summ = torch.logsumexp(post_loss, [0,1])
                    main_loss_summ = torch.logsumexp(main_loss, [0,1])
                else: post_loss_summ = post_loss; main_loss_summ = main_loss
                print0(f"kImg. : {kimg:.2f}, time : {timesec}, Post. loss : {post_loss_summ:.2f}, Main loss : {main_loss_summ:.2f}, Curr. loss : {loss}")
                # wandb.log({"post_loss" : post_loss_summ, "main_loss": main_loss_summ, "tot_loss": loss}, step=int(kimg))
            if (idx < save_freq and idx % (save_freq // 5) == 0) or (idx % save_freq == 0):
                kimg = idx*args.eff_batch_size // 1000
                for temp in [0.5, 0.7, 0.85, 1.0]:
                    x_samp = np.concatenate([main_model.sample(4, temp=temp) for _ in range(4)])
                    _ = utils.save_images(x_samp, f'{save_folder}/fakes_{kimg:06}_temp{temp}.png', imrange=[-0.5,0.5])
                    # wandb.log({f"main_example_{temp}": wandb.Image(f'{save_folder}/fakes_{kimg:06}_temp{temp}.png')}, step=kimg)
                xsamps = []
                for yt in y_tests:
                    # yt = yt.reshape(1,*yt.shape).to(device, non_blocking=True)
                    # yt = degradation.rev(yt)
                    # yt = torch.cat([yt]*4, dim=0)
                    x_samp = post_model.sample(4, yt, temp=1)
                    xsamps.append(x_samp)
                xsamps = np.concatenate(xsamps, axis=0)
                _ = utils.save_images(xsamps, f'{save_folder}/fakes_{kimg:06}_postsamp.png', imrange=[-0.5,0.5])
                # wandb.log({"post_example": wandb.Image(f'{save_folder}/fakes_{kimg:06}_postsamp.png')}, step=kimg)
                main_model.save(f'{save_folder}/main-network_{kimg:06}.pt')
                post_model.save(f'{save_folder}/post-network_{kimg:06}.pt')

        idx += 1
        if idx >= args.num_iters:
            break
        # torch.cuda.synchronize()
        # print("Rest :", time.time() - t)
        # t = time.time()


if __name__ == '__main__':

    args = parser.parse_args()

    args.world_size = args.num_gpus * args.num_nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # Effective batch size
    args.eff_batch_size = args.batch_size * args.world_size

    # importance weighting
    if args.importance_weighting: args.tiled_for_loss = 1

    # Spawn multiprocesses (one on each gpu)
    print(args)
    if args.world_size > 1:
        # mp.spawn(train, nprocs=args.num_gpus, args=(args,))
        train(args.rank, args)
    else:
        train(0, args)