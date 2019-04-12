#!/bin/bash
nvprof --track-memory-allocations on --profile-from-start off --profile-child-processes -fo %p.nvprof python3 model_interference_test.py