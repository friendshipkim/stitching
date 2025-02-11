import os
import numpy as np
import random
import torch
import argparse
from datasets import load_dataset
from evaluate import load

from transformers import TrainingArguments, Trainer, BertForSequenceClassification
from load_models import load_tokenizer, load_model

# default config
import config as cfg

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], return_tensors="pt", padding=True, truncation=True)
    return tokenizer(
        examples[sentence1_key], examples[sentence2_key], return_tensors="pt", padding=True, truncation=True
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a bert model on GLUE tasks")

    parser.add_argument(
        "--do_train",
        type=str2bool,
        nargs="?",
        const=True,
        default=cfg.do_train,
        help=f"Whether to train the model (Default: {cfg.do_train})",
    )
    parser.add_argument(
        "--use_wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=cfg.use_wandb,
        help=f"Whether to use wandb for logging (Default: {cfg.use_wandb})",
    )
    parser.add_argument("--seed", type=int, default=cfg.seed, help=f"Random seed, Default: {cfg.seed}")

    # huggingface trainer use gpu by default
    # parser.add_argument("--devid", type=int, default=cfg.seed, help=f"GPU id, if -1 use cpup (Default: {cfg.devid}")

    # dataset / model configs
    parser.add_argument(
        "--task",
        type=str,
        default=cfg.task,
        help=f"GLUE task name. Default: '{cfg.task}'",
    )
    parser.add_argument(
        "--src_model_name",
        type=str,
        default=cfg.src_model_name,
        help=f"Source model to load from huggingface model hub. Default: '{cfg.src_model_name}'",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=cfg.model_dir,
        help=f"Path to save finetuned models. Default: '{cfg.model_dir}'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=cfg.model_name,
        help=f"Model name to save. Default: '{cfg.model_name}'",
    )
    parser.add_argument(
        "--do_stitch",
        type=str2bool,
        nargs="?",
        const=True,
        default=cfg.do_stitch,
        help=f"Whether to finetine a stitched model. Default: {cfg.do_stitch}",
    )
    parser.add_argument(
        "--skip_layernorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=cfg.skip_layernorm,
        help=f"If stitch, whether to skip copying layernorm params. Default: {cfg.skip_layernorm}",
    )
    parser.add_argument(
        "--stitch_dummy",
        type=str2bool,
        nargs="?",
        const=True,
        default=cfg.stitch_dummy,
        help=f"Whether to stitch a dummy model initialized with eps or Xavier. Default: {cfg.stitch_dummy}",
    )

    # huggingface trainer args
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default=cfg.evaluation_strategy,
        help=f"The evaluation strategy to adopt during training. Possible values are: 'no', 'steps', 'epoch'. Default: '{cfg.evaluation_strategy}'",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default=cfg.save_strategy,
        help=f"The checkpoint save strategy to adopt during training. Possible values are: 'no', 'steps', 'epoch'. Default: '{cfg.save_strategy}'",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=cfg.learning_rate,
        help=f"The initial learning rate for AdamW optimizer. Default: {cfg.learning_rate}",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=cfg.per_device_train_batch_size,
        help=f"The batch size per GPU/TPU core/CPU for training. Default: {cfg.per_device_train_batch_size}",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=cfg.per_device_eval_batch_size,
        help=f"The batch size per GPU/TPU core/CPU for evaluation. Default: {cfg.per_device_eval_batch_size}",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int,
        default=cfg.num_train_epochs,
        help=f"Total number of training epochs to perform. Default: {cfg.num_train_epochs}"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=cfg.weight_decay,
        help=f"The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer. Default: {cfg.weight_decay}",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        type=str2bool,
        nargs="?",
        const=True,
        default=cfg.load_best_model_at_end,
        help=f"Whether or not to load the best model found during training at the end of training. Default: {cfg.load_best_model_at_end}",
    )
    parser.add_argument(
        "--src_model_dir2",
        type=str,
        default=None,
        help=f"Second model to stitch",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=cfg.epsilon,
        help=f"The pointwise std of the (normal) intialization of the cross diagonal terms. Default: {cfg.epsilon}",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # device = f"cuda:{args.devid}" if args.devid != -1 else "cpu"

    # ====== set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ====== dataset, metric, tokenizer
    task = args.task
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load("glue", actual_task)
    tokenizer = load_tokenizer(args.src_model_name)

    # ====== preprocess datasets
    sentence1_key, sentence2_key = task_to_keys[task]
    pre_tokenizer_columns = set(dataset["train"].features)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    tokenizer_columns = list(set(encoded_dataset["train"].features) - pre_tokenizer_columns)

    print("Columns added by tokenizer:", tokenizer_columns)

    # ====== num_labels, metric
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    validation_key = (
        "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    )

    # ====== define train args
    output_dir = os.path.join(args.model_dir, f"{args.model_name}_finetuned_{args.task}")
    best_model_path = os.path.join(output_dir, "checkpoint-best")

    train_args = {
        "output_dir": output_dir,
        "metric_for_best_model": metric_name,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": args.load_best_model_at_end,
        "report_to": "wandb" if args.use_wandb and args.do_train else "none",
    }

    # ====== train model
    if args.do_train:
        # # use wandb for logging
        # train_args["report_to"] = "none"

        # ====== load initialized models
        model = load_model(
            args.src_model_name,
            args.do_stitch,
            args.skip_layernorm,
            args.stitch_dummy,
            num_labels,
            src_model_dir2=args.src_model_dir2,
            epsilon=args.epsilon,
        )

        # pass args to trainer
        trainer = Trainer(
            model,
            args=TrainingArguments(**train_args),
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        print("=" * 6 + f" Training {args.model_name} " + "=" * 6)
        trainer.train()

        print("=" * 6 + f" Evaluating {args.model_name} " + "=" * 6)
        trainer.evaluate()
        print("=" * 6)

        print("=" * 6 + f" Saving best model to {best_model_path} " + "=" * 6)
        model.save_pretrained(best_model_path)

    # ====== eval only
    else:
        # load best model
        model = BertForSequenceClassification.from_pretrained(best_model_path, num_labels=num_labels)

        # update args
        # train_args["report_to"] = "none"
        train_args["resume_from_checkpoint"] = best_model_path

        # pass to trainer
        trainer = Trainer(
            model,
            args=TrainingArguments(**train_args),
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        print("=" * 6 + f" Evaluating {args.model_name} " + "=" * 6)
        result_dict = trainer.evaluate()
        print("=" * 6)

        breakpoint()