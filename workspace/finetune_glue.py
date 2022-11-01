import os
import numpy as np
import random
import torch
from datasets import load_dataset
from evaluate import load

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from load_models import load_tokenizer, load_models

# ====== global parameters
task = "mnli"
seed = 0
model_checkpoint = "prajjwal1/bert-mini-mnli"
batch_size = 64
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], return_tensors="pt", padding=True, truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], return_tensors="pt", padding=True, truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # ====== random seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # ====== dataset, metric, tokenizer
    # dataset = load_dataset("ptb_text_only")
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load("glue", actual_task)
    tokenizer = load_tokenizer()

    # ====== preprocess datasets
    sentence1_key, sentence2_key = task_to_keys[task]
    pre_tokenizer_columns = set(dataset["train"].features)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    tokenizer_columns = list(set(encoded_dataset["train"].features) - pre_tokenizer_columns)

    print("Columns added by tokenizer:", tokenizer_columns)

    # ====== finetune
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

    # TODO: change num_labels=num_labels
    # when stitching two different models
    src1_model, src2_model, tgt_model, stitched_model = load_models(num_labels)

    models = {
        # "bert-mini-mnli": AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels),
        "bert-mini": src1_model,
        "bert-small": tgt_model,
        "bert-mini-stitched-initln": stitched_model
    }

    for model_name, model in models.items():
        
        # define args
        args = TrainingArguments(
            output_dir=f"./saved_models/{model_name}_finetuned_{task}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            report_to="wandb"
        )

        # pass to trainer
        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
   
        print("=" * 6 + f" Training {model_name} " + "=" * 6)
        trainer.train()
           
        print("=" * 6 + f" Evaluating {model_name} " + "=" * 6)
        trainer.evaluate()
        print("=" * 6)

        breakpoint()
