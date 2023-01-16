# testing script for bert_mini-mini

python finetune_glue.py \
    --do_train False \
    --use_wandb False \
    --seed 0 \
    --task mnli \
    --src_model_name google/bert_uncased_L-4_H-256_A-4 \
    --model_dir ./saved_models/ \
    --model_name bert_mini-mini \
    --do_stitch True \
    --skip_layernorm False \
    --stitch_dummy False \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 5 \
    --weight_decay 0.01 \
    --load_best_model_at_end True