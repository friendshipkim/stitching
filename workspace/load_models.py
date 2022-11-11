import os
import copy
from typing import Type

import transformers
from transformers import BertForSequenceClassification, BertTokenizer, BertModel, StitchedBertConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers.models.bert.stitch_utils import stitch


# ====== global variables
devid = 0
src_model_name = "prajjwal1/bert-mini"
compare_model_name = "prajjwal1/bert-small"

do_stitch = True
skip_layernorm = False
stitch_dummy = False

# vocabs are identical for small and large
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(ROOT_DIR, "./vocab.txt")
input_text1 = "Where is art?"
input_text2 = "Where and what is art?"


def print_model_cfg(model):
    print(f"- num_parameters: {model.num_parameters()}")
    print(f"- hidden_size: {model.config.hidden_size}")
    print(f"- num_attention_heads: {model.config.num_attention_heads}")
    print(f"- num_hidden_layers: {model.config.num_hidden_layers}")
    print()


def load_tokenizer(model_name: str, model_max_length: int = 512) -> Type[transformers.BertTokenizerFast]:
    """load tokenizer of the given model name

    Args:
        model_name (str): model to load from huggingface model hub
        model_max_length (int, optional): maximum input sequence length. Defaults to 512.

    Returns:
        transformers.BertTokenizerFast: bert tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length)
    # or use vocab path
    # tokenizer = BertTokenizer(vocab_path)
    return tokenizer


def load_model(
    src_model_name: str, do_stitch: bool, skip_layernorm: bool, stitch_dummy: bool, device: str, num_labels: int = 2
) -> Type[BertForSequenceClassification]:
    """load either source or stitched model

    Args:
        src_model_name (str): Source model to load from huggingface model hub
        do_stitch (bool): Whether to finetine a stitched model
        skip_layernorm (bool): If stitch, whether to skip copying layernorm params
        stitch_dummy (bool): Whether to stitch a dummy model initialized with eps or Xavier
        device (str): Device to load model, cuda or cpu
        num_labels (int): The number of labels for sequence classification. Defaults to 2.

    Returns:
        BertForSequenceClassification: source or stitched model
    """
    # load pretrained models
    src_model = AutoModelForSequenceClassification.from_pretrained(src_model_name, num_labels=num_labels).to(device)

    # print spirce model configs
    print("=== source model ===")
    print_model_cfg(src_model)

    if do_stitch:
        # two models to be stitched
        # TODO: replace with different bert models
        src1_model = src_model
        src2_model = None if stitch_dummy else copy.deepcopy(src_model)

        # stitched config / model
        stitched_config = StitchedBertConfig(**src_model.config.to_dict(), num_labels=num_labels)
        stitched_model = BertForSequenceClassification(stitched_config).to(device)

        # print stitched model configs
        print("=== stitched model ===")
        print(f"skip_layernorm: {skip_layernorm}")
        print(f"stitch_dummy: {stitch_dummy}")
        print_model_cfg(stitched_model)

        # if the two models are stitchable, stitch them
        stitch(src1_model, src2_model, stitched_model, skip_layernorm, device)
        return stitched_model

    else:
        return src_model


def load_all_models(
    src_model_name: str,
    compare_model_name: str,
    skip_layernorm: bool,
    stitch_dummy: bool,
    device: str,
    num_labels: int = 2,
):
    """load small / large / stitched models at once

    Args:
        src_model_name (str): Source model to load from huggingface model hub
        compare_model_name (str): Model to compare with a stitched model
        skip_layernorm (bool): If stitch, whether to skip copying layernorm params
        stitch_dummy (bool): Whether to stitch a dummy model initialized with eps or Xavier
        device (str): Device to load model, cuda or cpu
        num_labels (int): The number of labels for sequence classification. Defaults to 2.

    Returns:
        Tuple: src1_model, src2_model, compare_model, stitched_model
    """
    # load pretrained models
    src_model = AutoModelForSequenceClassification.from_pretrained(src_model_name, num_labels=num_labels).to(device)
    compare_model = AutoModelForSequenceClassification.from_pretrained(compare_model_name, num_labels=num_labels).to(
        device
    )

    # set configs to return intermediate outputs
    # NOTE: disable this to use hf trainer
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining.forward
    # for model in [small_model, large_model]:
    #     model.config.output_hidden_states = True
    #     model.config.output_attentions = True

    # print small/large model configs
    print("=== source model ===")
    print_model_cfg(src_model)

    print("=== compare model ===")
    print_model_cfg(compare_model)

    # two models to be stitched
    # TODO: replace with different bert models
    src1_model = src_model
    src2_model = None if stitch_dummy else copy.deepcopy(src_model)

    # stitched config / model
    stitched_config = StitchedBertConfig(**src_model.config.to_dict(), num_labels=num_labels)
    stitched_model = BertForSequenceClassification(stitched_config).to(device)

    # print stitched model configs
    print("=== stitched model ===")
    print(f"skip_layernorm: {skip_layernorm}")
    print(f"stitch_dummy: {stitch_dummy}")
    print_model_cfg(stitched_model)

    # if the two models are stitchable, stitch them
    stitch(src1_model, src2_model, stitched_model, skip_layernorm, device)

    return src1_model, src2_model, compare_model, stitched_model


if __name__ == "__main__":
    device = f"cuda:{devid}" if devid != -1 else "cpu"

    # load tokenizer
    tokenizer = load_tokenizer(src_model_name)

    # load each model
    model = load_model(src_model_name, do_stitch, skip_layernorm, stitch_dummy, device)

    # load all models at once
    src1_model, src2_model, compare_model, stitched_model = load_all_models(
        src_model_name, compare_model_name, skip_layernorm, stitch_dummy, device
    )
    breakpoint()

    # forward
    encoded_input = tokenizer(input_text1, input_text2, return_tensors="pt").to(device)

    # outputs (dict)
    # keys: ['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
    src1_out = src1_model(**encoded_input)
    src2_out = src2_model(**encoded_input)
    tgt_out = stitched_model(**encoded_input)
