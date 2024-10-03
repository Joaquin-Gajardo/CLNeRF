#!/bin/bash

export ROOT_DIR=dataset/WAT
# export CUDA_HOME=/usr/local/cuda-11.6
# export PATH=/usr/local/cuda-11.6/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

scene_name=breville
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=car_resized
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 16.0 --eval_lpips --vocab_size=5 

scene_name=community
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 32.0 --eval_lpips --vocab_size=10 

scene_name=dyson
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 8.0 --eval_lpips --vocab_size=10 

scene_name=grill_resized
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 16.0 --eval_lpips --vocab_size=5 

scene_name=kitchen
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=living_room
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=mac
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 8.0 --eval_lpips --vocab_size=6 

scene_name=ninja
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 8.0 --eval_lpips --vocab_size=5 

scene_name=spa
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 16.0 --eval_lpips --vocab_size=5 

scene_name=street
python train_ngpa.py \
    --root_dir $ROOT_DIR/${scene_name} --dataset_name colmap_ngpa \
    --exp_name ${scene_name} --downsample 1.0 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --dim_a 48 --scale 32.0 --eval_lpips --vocab_size=5 
