#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python3 train_classifier.py -l 0.01
python3 train_classifier.py -l 0.05
python3 train_classifier.py -l 0.1
python3 train_classifier.py -l 0.9

python3 train_classifier.py -l 0.01 --dataset cifar --vaemodel models/VAE_cifar.cpkt
python3 train_classifier.py -l 0.05 --dataset cifar --vaemodel models/VAE_cifar.cpkt
python3 train_classifier.py -l 0.1 --dataset cifar --vaemodel models/VAE_cifar.cpkt
python3 train_classifier.py -l 0.9 --dataset cifar --vaemodel models/VAE_cifar.cpkt

python3 train_classifier.py -l 0.01 --dataset svhn --vaemodel models/VAE_svhn.cpkt
python3 train_classifier.py -l 0.05 --dataset svhn --vaemodel models/VAE_svhn.cpkt
python3 train_classifier.py -l 0.1 --dataset svhn --vaemodel models/VAE_svhn.cpkt
python3 train_classifier.py -l 0.9 --dataset svhn --vaemodel models/VAE_svhn.cpkt
