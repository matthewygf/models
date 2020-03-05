#!/bin/bash

# run at the run_imagenet.py dir
sudo docker run --rm -it --gpus '"device=1"' -v ${PWD}:/tfmodels nvcr.io/nvidia/tensorflow:19.10-py3 /bin/bash
