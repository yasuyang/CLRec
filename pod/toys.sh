#!/bin/bash
#SBATCH -J whole_sentence                # 作业名
#SBATCH -p Nvidia_A800
#SBATCH -N 1                            # 节点数量
#SBATCH -n 16                            # 申请的核心数
#SBATCH --gres=gpu:2                    # 每个节点申请的dcu数量
#SBATCH -o slurm-%j                     # 作业输出
#SBATCH -e slurm-%j                     # 作业输出

module load cuda/11.8

source ~/.bashrc 
conda activate podmaster


CUDA_VISIBLE_DEVICES=1 python -u pretrain.py \
--data_dir ./data/toys/ \
--model ./checkpoint/toys/checkpoint_com_exp/ \
--cuda \
--batch_size 64 \
--checkpoint ./checkpoint_trained/toys/checkpoint_com_exp/ \
--lr 0.0005

CUDA_VISIBLE_DEVICES=1 python -u seq.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 64 \
--checkpoint ./checkpoint_trained/toys/checkpoint_com_exp/

CUDA_VISIBLE_DEVICES=1 python -u topn.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 64 \
--checkpoint ./checkpoint_trained/toys/checkpoint_com_exp/
#
CUDA_VISIBLE_DEVICES=1 python -u exp.py \
--data_dir ./data/toys/ \
--cuda \
--batch_size 64 \
--checkpoint ./checkpoint_trained/toys/checkpoint_com_exp/