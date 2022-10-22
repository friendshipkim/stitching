"""
test stitching utils except for copy_layernorm with random inputs
"""
import torch
from torch import nn

import pytest

from transformers.models.bert.stitch_utils import copy_linear, copy_self_attn, copy_attention, copy_embeddings
from transformers.models.bert.modeling_bert import BertSelfAttention, BertEmbeddings, BertAttention


# ====== test functions
@pytest.mark.ffn
def test_embeddings(src_cfg, stitched_cfg, input_id_sample):
    # if test mode is in cfg, set true in bertconfig?
    # instead of passing test_mode
    src1_emb = BertEmbeddings(src_cfg)
    src2_emb = BertEmbeddings(src_cfg)
    tgt_emb = BertEmbeddings(stitched_cfg)

    copy_embeddings(src1_emb, src2_emb, tgt_emb)

    emb_input = {"input_ids": input_id_sample, "test_mode": True}

    src1_emb_out = src1_emb(**emb_input)
    src2_emb_out = src2_emb(**emb_input)
    tgt_emb_out = tgt_emb(**emb_input)

    assert torch.isclose(torch.cat((src1_emb_out, src2_emb_out), dim=-1), tgt_emb_out).all().item()


@pytest.mark.ffn
def test_linear(src_cfg, stitched_cfg, hidden_state_samples):
    src1_hidden, src2_hidden, tgt_hidden = hidden_state_samples

    src1_linear = nn.Linear(src_cfg.hidden_size, src_cfg.hidden_size)
    src2_linear = nn.Linear(src_cfg.hidden_size, src_cfg.hidden_size)
    tgt_linear = nn.Linear(stitched_cfg.hidden_size, stitched_cfg.hidden_size)

    copy_linear(src1_linear, src2_linear, tgt_linear, stitched_cfg.epsilon)

    src1_out = src1_linear(src1_hidden)
    src2_out = src2_linear(src2_hidden)
    tgt_out = tgt_linear(tgt_hidden)

    assert torch.isclose(torch.cat((src1_out, src2_out), dim=-1), tgt_out).all().item()


@pytest.mark.ffn
def test_copy_self_attn(src_cfg, stitched_cfg, hidden_state_samples):
    src1_hidden, src2_hidden, tgt_hidden = hidden_state_samples

    src1_self_attn = BertSelfAttention(src_cfg)
    src2_self_attn = BertSelfAttention(src_cfg)
    stitched_self_attn = BertSelfAttention(stitched_cfg)

    copy_self_attn(src1_self_attn, src2_self_attn, stitched_self_attn, stitched_cfg.epsilon)

    # NOTE: BertSelfAttention module outputs a tuple
    src1_out = src1_self_attn(src1_hidden, test_mode=True)[0]
    src2_out = src2_self_attn(src2_hidden, test_mode=True)[0]
    tgt_out = stitched_self_attn(tgt_hidden, test_mode=True)[0]

    assert torch.isclose(torch.cat((src1_out, src2_out), dim=-1), tgt_out).all().item()


@pytest.mark.ffn
def test_attn(src_cfg, stitched_cfg, hidden_state_samples):
    src1_hidden, src2_hidden, tgt_hidden = hidden_state_samples

    src1_attn = BertAttention(src_cfg)
    src2_attn = BertAttention(src_cfg)
    stitched_attn = BertAttention(stitched_cfg)

    copy_attention(src1_attn, src2_attn, stitched_attn, stitched_cfg.epsilon)

    # NOTE: BertAttention module outputs a tuple
    src1_out = src1_attn(src1_hidden, test_mode=True)[0]
    src2_out = src2_attn(src2_hidden, test_mode=True)[0]
    tgt_out = stitched_attn(tgt_hidden, test_mode=True)[0]

    assert torch.isclose(torch.cat((src1_out, src2_out), dim=-1), tgt_out).all().item()
