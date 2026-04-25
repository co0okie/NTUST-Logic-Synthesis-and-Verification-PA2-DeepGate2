#!/bin/bash

set -e

NUM_PROC=1
GPUS=0
EXP_ID=ablation_no_orthogonal_pi
MASTER_PORT=29512

cd src

echo "========================================"
echo "     Starting Stage 1 (No Ortho PI)     "
echo "========================================"
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id $EXP_ID \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --Prob_weight 1 --RC_weight 0 --Func_weight 0 \
 --num_rounds 1 \
 --gpus ${GPUS} --batch_size 16 \
 --no_rc \
 --disable_encode

echo "========================================"
echo "         Resetting Optimizer LR         "
echo "========================================"
python3 reset_pth.py prob --exp_id $EXP_ID --disable_encode

echo "========================================"
echo "     Starting Stage 2 (No Ortho PI)     "
echo "========================================"
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id $EXP_ID \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --Prob_weight 3 --RC_weight 1 --Func_weight 2 \
 --num_rounds 1 \
 --gpus ${GPUS} --batch_size 16 \
 --resume \
 --no_rc \
 --disable_encode

echo "========================================"
echo "         Ablation No Ortho Done!        "
echo "========================================"