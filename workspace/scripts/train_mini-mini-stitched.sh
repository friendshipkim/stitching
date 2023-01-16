#! /usr/bin/env bash

# training script for bert_mini

OTHER_PARAMS=${@:1}
export seed=$(($RANDOM%1000))

/home/me.docker/.conda/envs/pl/bin/python ~/work/stitching/workspace/finetune_glue.py \
    --do_train True \
    --use_wandb True \
    --seed $seed \
    --task mnli \
    --src_model_name google/bert_uncased_L-4_H-256_A-4 \
    --model_dir ./saved_models/ \
    --model_name bert_mini_seed_$seed \
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
    --epsilon 0.001 \
    $OTHER_PARAMS