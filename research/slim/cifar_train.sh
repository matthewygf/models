#!/bin/bash
echo "starting cifar 10"
python3 train_image_classifier.py --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir= \
    --model_name=resnet_v2_50
