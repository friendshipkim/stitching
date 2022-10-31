import os
import torch
import copy

from transformers import BertTokenizer, AutoModel, BertModel, StitchedBertConfig
from transformers.models.bert.stitch_utils import check_if_stitchable, stitch


# ====== global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
small_model_name = "prajjwal1/bert-mini"
large_model_name = "prajjwal1/bert-small"

# vocabs are identical for small and large
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(ROOT_DIR, "./vocab.txt")
input_text = "Example input."


def print_model_cfg(model):
    print(f"- num_parameters: {model.num_parameters()}")
    print(f"- hidden_size: {model.config.hidden_size}")
    print(f"- num_attention_heads: {model.config.num_attention_heads}")
    print(f"- num_hidden_layers: {model.config.num_hidden_layers}")
    print()


if __name__ == "__main__":
    # load pretrained models
    small_model = AutoModel.from_pretrained(small_model_name).to(device)
    large_model = AutoModel.from_pretrained(large_model_name).to(device)

    # set configs to return intermediate outputs
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining.forward
    for model in [small_model, large_model]:
        model.config.output_hidden_states = True
        model.config.output_attentions = True

    # print small/large model configs
    print("=== small model ===")
    print_model_cfg(small_model)

    print("=== large model ===")
    print_model_cfg(large_model)

    # two models to be stitched
    # TODO: replace with different bert models
    src1_model = small_model
    src2_model = copy.deepcopy(small_model)

    # stitched config / model
    stitched_config = StitchedBertConfig(**small_model.config.to_dict())
    stitched_model = BertModel(stitched_config).to(device)

    # print stitched model configs
    print("=== stitched model ===")
    print_model_cfg(stitched_model)

    # if the two models are stitchable, stitch them
    check_if_stitchable(src1_model.config, src2_model.config)
    stitch(src1_model, src2_model, stitched_model)

    # forward
    tokenizer = BertTokenizer(vocab_path)
    encoded_input = tokenizer(input_text, return_tensors="pt").to(device)

    # outputs (dict)
    # keys: ['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
    src1_out = src1_model(**encoded_input)
    src2_out = src2_model(**encoded_input)
    tgt_out = stitched_model(**encoded_input)

    breakpoint()
