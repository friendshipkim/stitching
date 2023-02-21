# Stitching Language Models

## 0. Create a conda environment
```
conda create -n stitching python=3.8
conda activate stitching
```

## 1. Install requirements & transformers
```
pip install -r requirements.txt
pip install -e .
```

## 2. Finetune models on GLUE datasets
### 1. Setup wandb
* [https://docs.wandb.ai/guides/integrations/huggingface](https://docs.wandb.ai/guides/integrations/huggingface)

### 2. Run finetuning
```
cd workspace

# Example) train `bert_mini-mini`
bash ./scripts/train_mini-mini.sh
```

* You can pass the arguments or change default values in `config.py` 
* See `dev_note.md` for the list of models
