""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import argparse
import models2
import os
import imageio as io

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--gen_dir", type=str, default='')
parser.add_argument("--num_images", type=int, default=10000)
parser.add_argument("--iter", type=str, default=20000)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--ambient", type=int, default=0)
parser.add_argument("--save_as", type=str, default='png')
parser.add_argument("--range", type=int, nargs='+', default=[-0.5,0.5])
parser.add_argument("--batch_size", type=int, default=10)
args = parser.parse_args()
print(args)

if args.ambient:
    args.iter = f'{int(args.iter):06}'
    model_path = os.path.join(args.gen_dir, f'main-network_{args.iter}.pt')
else:
    model_path = os.path.join(args.gen_dir, f'network_{args.iter}.pt')
model = models2.load_model(model_path, ambient=args.ambient, load_post_model=False).to(device)
fake_data = []
bs = args.batch_size
for i in range(int(np.ceil(args.num_images / bs))):
    fd = model.sample(bs, temp=args.temperature)
    fake_data.append(fd)
    print(i)
fake_data = np.concatenate(fake_data)
fake_data = fake_data[:args.num_images]
fake_data = (fake_data - args.range[0]) / (args.range[1] - args.range[0])

if args.save_as == 'npy':
    np.save(os.path.join(args.gen_dir, f'fake-{args.iter:06}-{args.temperature}-{args.num_images}.npy'), fake_data)

elif args.save_as == 'png':
    # process data to save as images
    fake_data = np.squeeze(fake_data)
    if len(fake_data.shape) == 4:
        fake_data = np.swapaxes(fake_data, 1, 2)
        fake_data = np.swapaxes(fake_data, 2, 3)
    fake_data = np.clip(fake_data, 0, 1)
    fake_data *= 255
    fake_data = fake_data.astype(np.uint8)
    fakes_folder = os.path.join(args.gen_dir, f'fake-{args.iter}-{args.temperature}-{args.num_images}')
    os.makedirs(fakes_folder, exist_ok=True)
    for i in range(args.num_images):
        io.imsave(
            os.path.join(fakes_folder, f'img_{i:05}.png'),
            fake_data[i],
        )
    # np.save(os.path.join(args.gen_dir, f'fake-{args.iter:06}-{args.temp}-{args.num_images}.npy'), fake_data)


    