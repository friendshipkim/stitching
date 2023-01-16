import os
import copy
import torch

import pytest
from typing import Type, Tuple

from transformers import AutoModel, BertModel, BertConfig, StitchedBertConfig, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding

# ====== global variables
bsz = 32
seq_len = 100
model_name = "google/bert_uncased_L-4_H-256_A-4"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(ROOT_DIR, "../vocab.txt")
input_text = "Example input."


# ====== fixture functions for configs / models
@pytest.fixture
def src_model(model_name: str = model_name) -> Type[BertModel]:
    # load small pretrained model
    src_model = AutoModel.from_pretrained(model_name)
    src_model.config.output_hidden_states = True
    src_model.config.output_attentions = True
    return src_model


@pytest.fixture
def src_cfg(src_model) -> Type[BertConfig]:
    return src_model.config


@pytest.fixture
def stitched_cfg(src_cfg) -> Type[StitchedBertConfig]:
    return StitchedBertConfig(**src_cfg.to_dict())


@pytest.fixture
def stitched_model(stitched_cfg) -> Type[BertModel]:
    return BertModel(stitched_cfg)


@pytest.fixture
def models(src_model, stitched_model) -> Tuple[Type[BertModel], Type[BertModel], Type[BertModel]]:
    # 2nd source model - copy src_model or define new one with src_config
    # return src_model, copy.deepcopy(src_model), stitched_model
    return src_model, BertModel(src_model.config), stitched_model


# ====== fixture functions for input/hidden states samples
@pytest.fixture()
def input_id_sample(src_cfg) -> Type[torch.LongTensor]:
    return torch.randint(0, src_cfg.vocab_size, (bsz, seq_len)).long()


@pytest.fixture()
def input_sample(vocab_path: str = vocab_path, input_text: str = input_text) -> Type[BatchEncoding]:
    tokenizer = BertTokenizer(vocab_path)
    encoded_input = tokenizer(input_text, return_tensors="pt")

    return encoded_input


@pytest.fixture()
def hidden_state_samples(
    src_cfg,
) -> Tuple[Type[torch.FloatTensor], Type[torch.FloatTensor], Type[torch.FloatTensor]]:
    src1_hidden = torch.rand((bsz, seq_len, src_cfg.hidden_size))
    src2_hidden = torch.rand((bsz, seq_len, src_cfg.hidden_size))
    tgt_hidden = torch.cat((src1_hidden, src2_hidden), dim=-1)

    return src1_hidden, src2_hidden, tgt_hidden
