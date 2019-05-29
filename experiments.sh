#!/bin/bash

# XResNet 
# for i in $(seq 1 30)
# do
#     echo noalpha_nosa_exp1 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=5 \
#         --size=128 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34 \
#         --alpha_pool=0 \
#         --logdir=noalpha_nosa_exp1
#     sleep 30
# done

# for i in $(seq 1 30)
# do
#     echo noalpha_nosa_exp2 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=5 \
#         --size=256 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34 \
#         --alpha_pool=0 \
#         --logdir=noalpha_nosa_exp2
#     sleep 30
# done

# for i in $(seq 1 30)
# do
#     echo noalpha_nosa_exp3 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=20 \
#         --size=256 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34 \
#         --alpha_pool=0 \
#         --logdir=noalpha_nosa_exp3
#     sleep 30
# done

# XResNet + SimpleSelfAttention 
# for i in $(seq 1 30)
# do
#     echo noalpha_sa_exp1 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=5 \
#         --size=128 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34_sa \
#         --alpha_pool=0 \
#         --logdir=noalpha_sa_exp1
#     sleep 30
# done

# for i in $(seq 1 30)
# do
#     echo noalpha_sa_exp2 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=5 \
#         --size=256 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34_sa \
#         --alpha_pool=0 \
#         --logdir=noalpha_sa_exp2
#     sleep 30
# done

# for i in $(seq 1 30)
# do
#     echo noalpha_sa_exp3 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=20 \
#         --size=256 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34_sa \
#         --alpha_pool=0 \
#         --logdir=noalpha_sa_exp3
#     sleep 30
# done

# XResNet + SimpleSelfAttention + AlphaPool
# for i in $(seq 1 30)
# do
#     echo alpha_sa_exp1 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=5 \
#         --size=128 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34_sa \
#         --alpha_pool=1 \
#         --logdir=alpha_sa_exp1
#     sleep 30
# done

# for i in $(seq 1 30)
# do
#     echo alpha_sa_exp2 loop: "$i"
#     python train_script.py --gpu=0 \
#         --epochs=5 \
#         --size=256 \
#         --woof=1 \
#         --bs=64 \
#         --lr=1e-2 \
#         --mixup=0.2 \
#         --fp16=1 \
#         --label_smooth=1 \
#         --arch_name=xresnet34_sa \
#         --alpha_pool=1 \
#         --logdir=alpha_sa_exp2
#     sleep 30
# done

for i in $(seq 1 30)
do
    echo alpha_sa_exp3 loop: "$i"
    python train_script.py --gpu=0 \
        --epochs=20 \
        --size=256 \
        --woof=1 \
        --bs=32 \
        --lr=1e-2 \
        --mixup=0.2 \
        --fp16=1 \
        --label_smooth=1 \
        --arch_name=xresnet34_sa \
        --alpha_pool=1 \
        --logdir=alpha_sa_exp3
    sleep 30
done