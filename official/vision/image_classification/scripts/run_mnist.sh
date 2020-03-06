#!/bin/sh
basic_cmd="python mnist_main.py --data_dir=/home/user/datasets/mnist --distribution_strategy=one_device --download --model_dir=/home/user/models_ckpt/mnist"
metrics_collection="sm__warps_active,dram__bytes_read,dram__bytes_write,smsp__inst_executed,smsp__cycles_active,sm__cycles_active,sm__issue_active,sm__inst_executed"
metrics_args="--metrics="$metrics_collection
sections_args="--section=ComputeWorkloadAnalysis --section=MemoryWorkloadAnalysis --section=Occupancy"
base_nv_cmd="nv-nsight-cu-cli --target-processes=all --profile-from-start=off --csv"

if [ -z "$1" ]
then
  echo "Not profiling";
  $basic_cmd --train_epochs=10
else
  echo "Will profile";
  if [ -z "$2" ]
  then
    echo $base_nv_cmd $basic_cmd --train_epochs=1
    $base_nv_cmd $basic_cmd --train_epochs=1
  else
    if [ -z "$3" ]
    then
      echo $base_nv_cmd $sections_args $basic_cmd --train_epochs=1
      $base_nv_cmd $sections_args $basic_cmd --train_epochs=1
    else
      echo $base_nv_cmd $sections_args $basic_cmd --train_epochs=1
      $base_nv_cmd $sections_args $basic_cmd --train_epochs=1 | tee $3 
    fi
  fi
fi
