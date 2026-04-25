#!/bin/bash

set -e

NUM_PROC=1
GPUS=0
DIM_HIDDEN=${1:-64}
MASTER_PORT=${2:-29500}
EXP_ID=dim$DIM_HIDDEN

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd src

python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id $EXP_ID \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --dim_hidden $DIM_HIDDEN \
 --Prob_weight 1 --RC_weight 0 --Func_weight 0 \
 --num_rounds 1 \
 --gpus ${GPUS} --batch_size 16 \
 --no_rc


python3 reset_pth.py prob --exp_id $EXP_ID --dim_hidden $DIM_HIDDEN


python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id $EXP_ID \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --dim_hidden $DIM_HIDDEN \
 --Prob_weight 3 --RC_weight 1 --Func_weight 2 \
 --num_rounds 1 \
 --gpus ${GPUS} --batch_size 16 \
 --resume \
 --no_rc