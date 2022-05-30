#!/bin/bash
#SBATCH -p tesla
#SBATCH -t 2-00:00:00
#SBATCH --gres gpu:4
#SBATCH -c 8
#SBATCH --mem-per-cpu=4000
#SBATCH -o ./output-%A.out
#SBATC -a 1-6

export OMP_NUM_THREADS=8
export TMP_DIR=/mnt/storage_2/project_data/grant_505/tmpdata

python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
    --num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
    --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 4 \
    --ch_mult 1 2 2 2 --save_content