#!/bin/bash
/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --o --f --profile-from-start=off --target-processes=all ./test.sh
