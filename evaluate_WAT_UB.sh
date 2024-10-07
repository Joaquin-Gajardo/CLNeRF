#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

render_scene() {
    local scene_name=$1
    local scale=$2
    local vocab_size=$3
    local checkpoint=$4
    local model=$5

    # Compute task_number and task_curr
    local task_number=$vocab_size
    local task_curr=$((vocab_size - 1))
    dataset="colmap_ngpa"
    results_file="results/$dataset/$model/$scene_name/test_results.json"
    if [ -f $results_file ]; then
        echo "Results file exists, skipping $scene_name"
        return
    fi
    
    echo "Evaluating $scene_name"

    python render_UB_WAT.py \
        --root_dir "dataset/WAT/${scene_name}" \
        --dataset_name $dataset \
        --exp_name ${scene_name} \
        --downsample 1.0 \
        --num_epochs 1 \
        --batch_size 8192 \
        --lr 1e-2 \
        --eval_lpips \
        --task_curr ${task_curr} \
        --task_number ${task_number} \
        --dim_a 48 \
        --dim_g 16 \
        --scale ${scale} \
        --vocab_size ${vocab_size} \
        --weight_path "ckpts/$model/$dataset/${scene_name}/${checkpoint}" \
        --render_fname "UB" \
        --val_only \
        --model $model \
        --no_save_test
}

model="NGPGv2"
echo "Rendering NGPGv2"
render_scene "breville" "8.0" "5" "epoch=19-v6.ckpt" $model
render_scene "car_resized" "16.0" "5" "epoch=19-v1.ckpt" $model
render_scene "community" "32.0" "10" "epoch=19.ckpt" $model
render_scene "dyson" "8.0" "5" "epoch=19.ckpt" $model
render_scene "grill_resized" "16.0" "5" "epoch=19.ckpt" $model
render_scene "kitchen" "8.0" "5" "epoch=19.ckpt" $model
render_scene "living_room" "8.0" "5" "epoch=19.ckpt" $model
render_scene "mac" "8.0" "6" "epoch=19-v1.ckpt" $model
render_scene "ninja" "8.0" "5" "epoch=19.ckpt" $model
render_scene "spa" "16.0" "5" "epoch=19.ckpt" $model
render_scene "street" "32.0" "5" "epoch=19.ckpt" $model

model="NGPA"
echo "Rendering NGPA"
render_scene "breville" "8.0" "5" "epoch=19.ckpt" $model
render_scene "car_resized" "16.0" "5" "epoch=19.ckpt" $model
render_scene "community" "32.0" "10" "epoch=19.ckpt" $model
render_scene "dyson" "8.0" "5" "epoch=19-v1.ckpt" $model
render_scene "grill_resized" "16.0" "5" "epoch=19.ckpt" $model
render_scene "kitchen" "8.0" "5" "epoch=19.ckpt" $model
render_scene "living_room" "8.0" "5" "epoch=19.ckpt" $model
render_scene "mac" "8.0" "6" "epoch=19.ckpt" $model
render_scene "ninja" "8.0" "5" "epoch=19.ckpt" $model
render_scene "spa" "16.0" "5" "epoch=19.ckpt" $model
render_scene "street" "32.0" "5" "epoch=19.ckpt" $model

model="NGP"
echo "Rendering NGP"
render_scene "breville" "8.0" "5" "epoch=19.ckpt" $model
render_scene "car_resized" "16.0" "5" "epoch=19.ckpt" $model
render_scene "community" "32.0" "10" "epoch=19.ckpt" $model
render_scene "dyson" "8.0" "5" "epoch=19.ckpt" $model
render_scene "grill_resized" "16.0" "5" "epoch=19.ckpt" $model
render_scene "kitchen" "8.0" "5" "epoch=19.ckpt" $model
render_scene "living_room" "8.0" "5" "epoch=19.ckpt" $model
render_scene "mac" "8.0" "6" "epoch=19.ckpt" $model
render_scene "ninja" "8.0" "5" "epoch=19.ckpt" $model
render_scene "spa" "16.0" "5" "epoch=19.ckpt" $model
render_scene "street" "32.0" "5" "epoch=19.ckpt" $model