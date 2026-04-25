#!/bin/bash
NUM_PROC=1
GPUS=0
MASTER_PORT=${1:-29500}
DIM_HIDDEN=${2:-64}

export LD_LIBRARY_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH"

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id dim$DIM_HIDDEN \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --dim_hidden $DIM_HIDDEN \
 --Prob_weight 3 --RC_weight 1 --Func_weight 2 \
 --num_rounds 1 \
 --gpus ${GPUS} --batch_size 16 \
 --resume \
 --no_rc

