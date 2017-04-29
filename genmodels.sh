#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python3 train_vae.py --vaemodel models/VAE_mnist.cpkt
python3 train_vae.py --dataset cifar --datapath cifar10/cifar10.hdf5 --vaemodel models/VAE_cifar.cpkt
python3 train_vae.py --dataset svhn --datapath svhn --vaemodel models/VAE_svhn.cpkt
