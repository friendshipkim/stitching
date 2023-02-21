#! /usr/bin/env bash

# export WANDB_MODE=disabled

if [ "$#" -le 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$1
fi

echo Using GPU Device $CUDA_VISIBLE_DEVICES
export WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7
export seed=$(($RANDOM%1000))

local_dir=`dirname $0`
echo $local_dir


/home/me.docker/.conda/envs/pl/bin/python ~/work/stitching/workspace/finetune_glue.py \
    --do_train True \
    --use_wandb True \
    --seed $seed \
    --task mnli \
    --src_model_name $local_dir/saved_models/bert_mini_seed_856_finetuned_mnli/checkpoint-61360// \
    --model_dir $local_dir/saved_models/ \
    --model_name bert_mini-mini-stitched-noisyfinetune_$seed \
    --do_stitch True \
    --skip_layernorm False \
    --stitch_dummy False \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --load_best_model_at_end True \
    --src_model_dir2 $local_dir/saved_models/bert_mini_seed_90_finetuned_mnli/checkpoint-best/ \
    --epsilon 0.001