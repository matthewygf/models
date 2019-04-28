#!/bin/bash
timestamp=$(date +%s)
/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu-cli --o=$timestamp --f --profile-from-start=off --target-processes=all ./test.sh
