#!/bin/bash
set -o errexit -o pipefail -o noclobber -o nounset

! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
  echo ''getopt --test' failed in this environment'
  exit 1
fi

OPTIONS=scp:m:b:
LONGOPS=section,csv,prof-option:,model:,batchsize:

! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
   exit 2
fi

eval set -- "$PARSED"

s=n c=n profOption=- model="resnet50_v1.5" batchsize=8

while true; do
  echo $1
  case "$1" in
    -s|--section)
      s=y
      shift
      ;;
    -c|--csv)
      c=y
      shift
      ;;
    -p|--prof-option)
      profOption="$2"
      shift 2
      ;;
    -b|--batchsize)
      batchsize="$2"
      shift 2
      ;;
    -m|--model)
      model="$2"
      shift 2
      ;;
    --)
      break 1 
      ;;
    *)
      echo "Unknown param: $1"
      exit 3
      ;;
  esac
done

if [[ $# -ne 1 ]]; then
   echo "$0: require profile option"
   exit 4
fi

echo "section: $s, csv: $c, prof option: $profOption, model: $model batch: $batchsize"

vision_kernels="(?i).*dnn.*|.*conv.*|.*gemm.*|.*grad.*|.*fft.*|.*im2col.*|.*transpose.*|.*mult.*"
basic_cmd="python -m resnet_imagenet_main --model_dir=~/models_ckpt/imagenetsynth --use_synthetic_data=true --num_gpus=1 --batch_size=$batchsize --train_epochs=1 --train_steps=1 --skip_eval --log_steps=1 --model=$model"
sections_args="--section=ComputeWorkloadAnalysis --section=MemoryWorkloadAnalysis --section=Occupancy"
metrics_args="sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"
#metrics_args="sm__warps_active.avg.pct_of_peak_sustained_active"
metrics_combined="--metrics=$metrics_args"
base_nvnsight_cmd="nv-nsight-cu-cli --target-processes=all --profile-from-start=off --kernel-regex-base=function --kernel-regex=$vision_kernels"

base_nvprof_cmd="nvprof"

if [[ $profOption == "nvprof" ]]
then
  do_cmd=$base_nvprof_cmd
else
  do_cmd=$base_nvnsight_cmd
fi

if [ $s == "y" ]
then
  do_cmd="$do_cmd $metrics_combined"
fi 

iskeras=""
if [[ "$model" =~ ^(mobilenet|inceptionv3|nasnetmobile|mobilenetv2|densenet121|densenet169|densenet201|vgg) ]]; then
  iskeras="--keras_application_models"
fi

basic_cmd="$basic_cmd $iskeras"


if [ $c == "y" ]
then
  unixtimestamp=$(date +%s)
  logfile=prof_$model
  logfile="$logfile"-$unixtimestamp-b$batchsize.csv
  do_cmd="$do_cmd --csv" 
  if [[ $profOption == "nvprof" ]]
  then
    do_cmd="$do_cmd --log_file=$logfile $basic_cmd"
  else
    do_cmd="$do_cmd $basic_cmd"
  fi
else
  do_cmd="$do_cmd $basic_cmd"
fi

echo $do_cmd
$do_cmd | tee $logfile
