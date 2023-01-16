#! /usr/bin/env bash

# training script for bert_mini

if [ "$#" -le 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$1
fi

echo $CUDA_VISIBLE_DEVICES
# export WANDB_MODE=disabled
export WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7
export seed=$(($RANDOM%1000))

/home/me.docker/.conda/envs/pl/bin/python ~/work/stitching/workspace/finetune_glue.py \
    --do_train True \
    --use_wandb True \
    --seed $seed \
    --task mnli \
    --src_model_name google/bert_uncased_L-4_H-256_A-4 \
    --model_dir ./saved_models/ \
    --model_name bert_mini_seed_$seed \
    --do_stitch False \
    --skip_layernorm False \
    --stitch_dummy False \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --load_best_model_at_end True