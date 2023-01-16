from transformers import BertTokenizer, BertConfig
from pretraining.modeling import BertForSequenceClassification

model_paths = [
    "/n/home05/wk247/workspace/academic-budget-bert/saved_models/training-out-halflarge/halflarge_pretraining-0/0/epoch1000000_step10102",
    "/n/home05/wk247/workspace/academic-budget-bert/saved_models/training-out-halflarge/halflarge_pretraining-1/1/epoch1000000_step10010",
    "/n/home05/wk247/workspace/academic-budget-bert/saved_models/training-out-halflarge/halflarge_pretraining-2/2/epoch1000000_step9799",
    "/n/home05/wk247/workspace/academic-budget-bert/saved_models/training-out-halflarge/halflarge_pretraining-3/3/epoch1000000_step9923"   
]

num_labels = 2
config = BertConfig.from_pretrained(model_paths[0], num_labels=num_labels, layer_norm_type='pytorch')
tokenizer = BertTokenizer.from_pretrained(model_paths[0])
model = BertForSequenceClassification.from_pretrained(model_paths[0], config=config)
breakpoint()
