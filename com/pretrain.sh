#!/bin/bash
#SBATCH -J toys_com_exp                    # 作业名
#SBATCH -p Nvidia_A800
#SBATCH -N 1                            # 节点数量
#SBATCH -n 16                            # 申请的核心数
#SBATCH --gres=gpu:2                    # 每个节点申请的dcu数量
#SBATCH -o slurm-%j                     # 作业输出
#SBATCH -e slurm-%j                     # 作业输出

module load cuda/11.8

source ~/.bashrc 
conda activate podmaster

# 定义变量
BACKBONE="./t5_small"
TRAIN_PATH="./data/toys/exp_com_data/train_data.json"
VALID_PATH="./data/toys/exp_com_data/valid_data.json"
TASK="seq"
DATASET="toys"
CUTOFF=512
MODEL_DIR="./checkpoint/toys/checkpoint_exp"
BATCH_SIZE=64
VALID_SELECT=1
EPOCHS=4
LR=0.001
WARMUP_STEPS=100
LOGGING_STEPS=10
OPTIM="adamw_torch"
EVAL_STEPS=200
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3
is_whole_sentence=false

# 执行 Python 脚本
torchrun --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
  --backbone $BACKBONE \
  --train_path $TRAIN_PATH \
  --valid_path $VALID_PATH \
  --task $TASK \
  --dataset $DATASET \
  --cutoff $CUTOFF \
  --model_dir $MODEL_DIR \
  --batch_size $BATCH_SIZE \
  --valid_select $VALID_SELECT \
  --epochs $EPOCHS \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --logging_steps $LOGGING_STEPS \
  --optim $OPTIM \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --is_whole_sentence $is_whole_sentence