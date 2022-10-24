"""
test stitching utils except for copy_layernorm with random inputs
"""
import torch
from torch import nn

import pytest

from transformers.models.bert.stitch_utils import (
    copy_linear,
    copy_self_attn,
    copy_attention,
    copy_embeddings,
    copy_layer,
    stitch,
)
from transformers.models.bert.modeling_bert import BertSelfAttention, BertEmbeddings, BertAttention, BertLayer

# lower precision
atol = 1e-07


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


@pytest.mark.ffn
def test_layer(src_cfg, stitched_cfg, hidden_state_samples):
    src1_hidden, src2_hidden, tgt_hidden = hidden_state_samples

    src1_layer = BertLayer(src_cfg)
    src2_layer = BertLayer(src_cfg)
    stitched_layer = BertLayer(stitched_cfg)

    copy_layer(src1_layer, src2_layer, stitched_layer, stitched_cfg.epsilon)

    # NOTE: BertLayer module outputs a tuple
    src1_out = src1_layer(src1_hidden, test_mode=True)[0]
    src2_out = src2_layer(src2_hidden, test_mode=True)[0]
    tgt_out = stitched_layer(tgt_hidden, test_mode=True)[0]

    assert torch.isclose(torch.cat((src1_out, src2_out), dim=-1), tgt_out, atol=atol).all().item()


@pytest.mark.ffn
def test_bert(models, input_sample):
    src1_model, src2_model, stitched_model = models

    stitch(src1_model, src2_model, stitched_model)

    # outputs (dict)
    # keys: ['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
    src1_out = src1_model(**input_sample, test_mode=True)
    src2_out = src2_model(**input_sample, test_mode=True)
    tgt_out = stitched_model(**input_sample, test_mode=True)

    # check embeddings (hidden_states[0]) and hidden states (layer outputs)
    for src1_hidden, src2_hidden, tgt_hidden in zip(
        src1_out["hidden_states"], src2_out["hidden_states"], tgt_out["hidden_states"]
    ):
        assert torch.isclose(torch.cat((src1_hidden, src2_hidden), dim=-1), tgt_hidden).all()

    # check pooler output with lower precision
    assert torch.isclose(
        torch.cat((src1_out["pooler_output"], src2_out["pooler_output"]), dim=-1), tgt_out["pooler_output"], atol=atol
    ).all()

    # check attention scores
    for src1_attn_score, src2_attn_score, tgt_attn_score in zip(
        src1_out["attentions"], src2_out["attentions"], tgt_out["attentions"]
    ):
        assert torch.isclose(torch.cat((src1_attn_score, src2_attn_score), dim=1), tgt_attn_score).all()
