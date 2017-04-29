#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python3 train_classifier.py -l 0.01
python3 train_classifier.py -l 0.05
python3 train_classifier.py -l 0.1
python3 train_classifier.py -l 0.9



