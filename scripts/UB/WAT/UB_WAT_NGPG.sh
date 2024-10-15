#!/bin/bash

export ROOT_DIR=dataset/WAT
# export CUDA_HOME=/usr/local/cuda-11.6
# export PATH=/usr/local/cuda-11.6/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

model="NGPG"
exp="exp2" # exp0 was original split from name_to_task, exp1 we skipped it was a dim_g=32 experiment for NGPGv2 only
#scene_name=breville
#python train_ngpg.py \
#    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
#    --exp_name "$exp/${scene_name}" --downsample 1.0 \
#    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 --model $model

scene_name=car_resized
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 --model $model

scene_name=community
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 32.0 --eval_lpips --vocab_size=10 --model $model

scene_name=dyson
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 --model $model

scene_name=grill_resized
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 --model $model

scene_name=kitchen
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 --model $model

scene_name=living_room
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 --model $model

scene_name=mac
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=6 --model $model

scene_name=ninja
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 8.0 --eval_lpips --vocab_size=5 --model $model

scene_name=spa
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 16.0 --eval_lpips --vocab_size=5 --model $model

scene_name=street
python train_ngpg.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name "$exp/${scene_name}" --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_g 16 --scale 32.0 --eval_lpips --vocab_size=5 --model $model
