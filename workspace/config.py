"""
configurations
"""
do_train = True
use_wandb = True
seed = 0
# devid = 0  # gpu id: 'cpu' if -1

# ====== dataset / model configs
task = "mnli"
src_model_name = "google/bert_uncased_L-4_H-256_A-4"
# compare_model_name = "google/bert_uncased_L-4_H-512_A-8"

model_dir = "./saved_models/"
model_name = "bert_mini-mini"
do_stitch = True
skip_layernorm = False
stitch_dummy = False

# ====== trainer args
# reference: https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments
evaluation_strategy = "epoch"
save_strategy = "epoch"
learning_rate = 2e-5
per_device_train_batch_size = 64
per_device_eval_batch_size = 64
num_train_epochs = 5
weight_decay = 0.01
load_best_model_at_end = True
epsilon = 1e-3