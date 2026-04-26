#!/bin/bash

# 檢查輸入參數是否足夠
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <gpu_id>"
    echo "Available experiments: dim32, dim64, dim128, dim64_no_tt, dim64_no_orthogonal_pi"
    exit 1
fi

EXPERIMENT=$1
GPU_ID=$2

# 強制隔離 GPU (解決原作者程式碼 Bug)
export CUDA_VISIBLE_DEVICES=$GPU_ID
# 既然環境已經被隔離，傳給 Python 的 GPU 編號固定為 0
GPUS_ARG=0
NUM_PROC=1
EXP_ID="$EXPERIMENT"

# 根據 $1 設定對應的參數
case $EXPERIMENT in
    dim32)
        MASTER_PORT=29500
        DIM_HIDDEN=32
        DISABLE_ENCODE=""
        # Stage 2 權重 (Baseline 預設值)
        S2_PROB=3; S2_RC=1; S2_FUNC=2
        ;;
    dim64)
        MASTER_PORT=29501
        DIM_HIDDEN=64
        DISABLE_ENCODE=""
        # Stage 2 權重 (Baseline 預設值)
        S2_PROB=3; S2_RC=1; S2_FUNC=2
        ;;
    dim128)
        MASTER_PORT=29502
        DIM_HIDDEN=128
        DISABLE_ENCODE=""
        # Stage 2 權重 (Baseline 預設值)
        S2_PROB=3; S2_RC=1; S2_FUNC=2
        ;;
    dim64_no_tt)
        MASTER_PORT=29503
        DIM_HIDDEN=64
        DISABLE_ENCODE=""
        # Stage 2 權重 (Ablation: Func loss 設為 0)
        S2_PROB=3; S2_RC=1; S2_FUNC=0
        ;;
    dim64_no_orthogonal_pi)
        MASTER_PORT=29504
        DIM_HIDDEN=64
        # Ablation: 拔除正交編碼
        DISABLE_ENCODE="--disable_encode"
        # Stage 2 權重 (Baseline 預設值)
        S2_PROB=3; S2_RC=1; S2_FUNC=2
        ;;
    *)
        echo "Error: Unknown experiment '$EXPERIMENT'"
        exit 1
        ;;
esac

cd src

echo "========================================================="
echo " Starting Experiment: $EXPERIMENT on Physical GPU: $GPU_ID "
echo " Port: $MASTER_PORT | Dim: $DIM_HIDDEN | Encode: ${DISABLE_ENCODE:-Default} "
echo "========================================================="

echo ">>> [1/3] Running Stage 1 ..."
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id $EXP_ID \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --Prob_weight 1 --RC_weight 0 --Func_weight 0 \
 --num_rounds 1 \
 --gpus $GPUS_ARG \
 --batch_size 64 \
 --num_epochs 20 \
 --lr_step 15 \
 --no_rc \
 --dim_hidden $DIM_HIDDEN \
 $DISABLE_ENCODE

echo ">>> [2/3] Resetting Optimizer Learning Rate ..."
python3 reset_pth.py prob \
 --exp_id $EXP_ID \
 --dim_hidden $DIM_HIDDEN \
 $DISABLE_ENCODE

echo ">>> [3/3] Running Stage 2 ..."
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT ./main.py prob \
 --exp_id $EXP_ID \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --Prob_weight $S2_PROB --RC_weight $S2_RC --Func_weight $S2_FUNC \
 --num_rounds 1 \
 --gpus $GPUS_ARG \
 --batch_size 64 \
 --num_epochs 20 \
 --lr_step 15 \
 --resume \
 --no_rc \
 --dim_hidden $DIM_HIDDEN \
 $DISABLE_ENCODE

echo "========================================================="
echo " Experiment $EXPERIMENT Finished! "
echo "========================================================="
