#! /usr/bin/env bash
python /opt/ml/code/workspace/finetune_glue.py \
       --do_train True --seed 0 --devid 0 --task mnli --src_model_name prajjwal1/bert-mini --model_dir ./saved_models/ \
       --model_name bert_mini-mini --do_stitch True --skip_layernorm False --stitch_dummy False --evaluation_strategy epoch \
       --save_strategy epoch --learning_rate 2e-5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --num_train_epochs 5 \
       --weight_decay 0.01 --load_best_model_at_end True