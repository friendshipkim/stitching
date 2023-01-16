import os
from transformers import BertTokenizer, BertLMHeadModel

model_name = "bert-large-uncased"
# model_name = "google/bert_uncased_L-4_H-256_A-4" # bert-mini
# model_name = "google/bert_uncased_L-4_H-512_A-8" # bert-small
save_path = os.path.join("/n/home05/wk247/workspace/stitching/workspace/local_cache/", model_name)

save_tokenizer = True
save_model = True

# <class 'pretraining.modeling.BertLMHeadModel'> 
# <class 'pretraining.configs.PretrainedBertConfig'> 
# <class 'transformers.models.bert.tokenization_bert.BertTokenizer'>

# load tokenizer and save
tokenizer = BertTokenizer.from_pretrained(model_name)
if save_tokenizer:
    # save
    tokenizer.save_pretrained(save_path)
    print(f"{model_name} tokenizer saved to '{save_path}'")
    
    # check
    try:
        tokenizer_local = BertTokenizer.from_pretrained(save_path)
    except:
        print("tokenizer is not saved properly, exit")
        assert False



# load tokenizer and save
model = BertLMHeadModel.from_pretrained(model_name)
if save_model:
    model.save_pretrained(save_path)
    print(f"{model_name} model saved to '{save_path}'")
    
    # check
    try:
        model_local = BertLMHeadModel.from_pretrained(save_path)
    except:
        print("model is not saved properly, exit")
        assert False

breakpoint()