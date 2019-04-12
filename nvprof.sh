#!/bin/bash
nvprof --track-memory-allocations on --profile-from-start off --profile all processes -fo %p.nvprof 