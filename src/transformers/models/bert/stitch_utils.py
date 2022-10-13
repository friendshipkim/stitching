"""
util functions to stitch ffn layers
"""
import torch
from torch import nn
from typing import Type

from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertEmbeddings, BertAttention


# TODO: merge this into StitchedBertConfig
def check_if_stitchable(src1_cfg: Type[BertConfig], src2_cfg: Type[BertConfig]) -> None:
    """
    Given two bert configs, check if the two models are stitchable

    Args:
        src1_cfg (transformers.BertConfig): first source model config
        src2_cfg (transformers.BertConfig): second source model config
    """
    assert src1_cfg.vocab_size == src2_cfg.vocab_size, "vocab sizes should match"
    assert src1_cfg.num_hidden_layers == src2_cfg.config.num_hidden_layers, "number of hidden layers should match"


def copy_linear(src1: Type[nn.Linear], src2: Type[nn.Linear], tgt: Type[nn.Linear], epsilon: float) -> None:
    """
    Diagonally copy the weights of the two source Linear layers to the target layer.
    Set non-diagonal parts to epsilon

    Args:
        src1 (torch.nn.Linear): first source Linear layer
        src2 (torch.nn.Linear): second source Linear layer
        tgt (torch.nn.Linear): target Linear layer
        epsilon (float): float number to fill non-diagonal parts
    """
    # Check if bias exists
    assert None not in (src1.bias, src2.bias, tgt.bias) or not any((src1.bias, src2.bias, tgt.bias))

    src1_out_dim, src1_in_dim = src1.weight.size()
    src2_out_dim, src2_in_dim = src2.weight.size()
    tgt_out_dim, tgt_in_dim = tgt.weight.size()

    assert tgt_out_dim == src1_out_dim + src2_out_dim
    assert tgt_in_dim == src1_in_dim + src2_in_dim

    # Initialize with epsilon
    tgt.weight.data[:] = epsilon

    # Copy weights diagonally
    tgt.weight.data[:src1_out_dim, :src1_in_dim] = src1.weight.data
    tgt.weight.data[-src2_out_dim:, -src2_in_dim:] = src2.weight.data

    # If biases exist, copy biases
    if tgt.bias is not None:
        tgt.bias.data[:src1_out_dim] = src1.bias.data
        tgt.bias.data[-src2_out_dim:] = src2.bias.data


def copy_layernorm(src1: Type[nn.LayerNorm], src2: Type[nn.LayerNorm], tgt: Type[nn.LayerNorm]) -> None:
    """
    Copy the weights of the two source LayerNorm layers to the target layer

    Args:
        src1 (torch.nn.LayerNorm): first source LayerNorm
        src2 (torch.nn.LayerNorm): second source LayerNorm
        src1 (torch.nn.LayerNorm): target LayerNorm
    """
    src1_dim, src2_dim, tgt_dim = src1.weight.size(0), src2.weight.size(0), tgt.weight.size(0)
    assert tgt_dim == src1_dim + src2_dim

    # Copy weights
    tgt.weight.data[:src1_dim] = src1.weight.data
    tgt.weight.data[-src2_dim:] = src2.weight.data

    # Copy biases
    tgt.bias.data[:src1_dim] = src1.bias.data
    tgt.bias.data[-src2_dim:] = src2.bias.data


def copy_self_attn(
    src1: Type[BertSelfAttention], src2: Type[BertSelfAttention], tgt: Type[BertSelfAttention], epsilon: float
) -> None:
    """
    Copy the linear projections of the two source BertSelfAttention modules to the target module
    Set the rest to epsilon

    Args:
        src1 (transformers.models.bert.modeling_bert.BertSelfAttention): first source BertSelfAttention module
        src2 (transformers.models.bert.modeling_bert.BertSelfAttention): second source BertSelfAttention module
        tgt (transformers.models.bert.modeling_bert.BertSelfAttention): target BertSelfAttention module
        epsilon (float): float number to fill the rest
    """
    src1_dim, src2_dim, _ = src1.query.weight.size(0), src2.query.weight.size(0), tgt.query.weight.size(0)

    for transform_type in ["query", "key", "value"]:
        # Initialize with epsilon
        tgt.get_submodule(transform_type).weight.data[:] = epsilon

        # TODO: remove extra dimension, add `_copy_projection` func
        # Copy weights, src1 - top left, src2 - bottom right
        tgt.get_submodule(transform_type).weight.data[0, :src1_dim, :src1_dim] = src1.get_submodule(
            transform_type
        ).weight.data
        tgt.get_submodule(transform_type).weight.data[1, -src2_dim:, -src2_dim:] = src2.get_submodule(
            transform_type
        ).weight.data

        # Copy biases
        tgt.get_submodule(transform_type).bias.data[0, :src1_dim] = src1.get_submodule(transform_type).bias.data
        tgt.get_submodule(transform_type).bias.data[1, -src2_dim:] = src2.get_submodule(transform_type).bias.data


def copy_attentions(
    src1: Type[BertAttention], src2: Type[BertAttention], tgt: Type[BertAttention], epsilon: float
) -> None:
    """
    Copy input/output linear projections and layernorm of the two source BertAttention modules to the target module
    Set the rest to epsilon

    Args:
        src1 (transformers.models.bert.modeling_bert.BertAttention): first source BertAttention module
        src2 (transformers.models.bert.modeling_bert.BertAttention): second source BertAttention module
        tgt (transformers.models.bert.modeling_bert.BertAttention): target BertAttention module
        epsilon (float): float number to fill the rest
    """

    # Key, query, value projections
    copy_self_attn(src1.self, src2.self, tgt.self, epsilon)

    # Output projection
    # TODO: check output projections
    copy_linear(src1.output.dense, src2.output.dense, tgt.output.dense, epsilon)

    # Layernorm
    copy_layernorm(src1.output.LayerNorm, src2.output.LayerNorm, tgt.output.LayerNorm)


def copy_embeddings(src1: Type[BertEmbeddings], src2: Type[BertEmbeddings], tgt: Type[BertEmbeddings]) -> None:
    """
    Copy embeddings and layernorm of the two source BertEmbeddings modules to the target module

    Args:
        src1 (transformers.models.bert.modeling_bert.BertEmbeddings): first source BertEmbeddings module
        src2 (transformers.models.bert.modeling_bert.BertEmbeddings): second source BertEmbeddings module
        tgt (transformers.models.bert.modeling_bert.BertEmbeddings): target BertEmbeddings module
    """
    # Embeddings
    embed_types = ["word_embeddings", "position_embeddings", "token_type_embeddings"]
    for embed_type in embed_types:
        tgt.get_submodule(embed_type).weight.data[:] = torch.cat(
            (
                src1.get_submodule(embed_type).weight.data,
                src2.get_submodule(embed_type).weight.data,
            ),
            dim=-1,
        )

    # Layernorm
    copy_layernorm(src1.LayerNorm, src2.LayerNorm, tgt.LayerNorm)


def stitch(src1: Type[BertModel], src2: Type[BertModel], tgt: Type[BertModel]) -> None:
    """
    Stitch two Bert models by copying the internal weights

    Args:
        src1 (transformer.BertModel): two source models to stitch
        src2 (transformer.BertModel): two source models to stitch
        tgt (transformer.BertModel): stitched target model
    """
    epsilon = tgt.config.epsilon

    # Embeddings
    copy_embeddings(src1.embeddings, src2.embeddings, tgt.embeddings)

    # Copy transformer layers
    for layer_1, layer_2, layer_st in zip(src1.encoder.layer, src2.encoder.layer, tgt.encoder.layer):
        # Multihead attentions
        copy_attentions(layer_1.attention, layer_2.attention, layer_st.attention, epsilon)

        # Intermediate ffn
        copy_linear(layer_1.intermediate.dense, layer_2.intermediate.dense, layer_st.intermediate.dense, epsilon)

        # Output ffn
        copy_linear(layer_1.output.dense, layer_2.output.dense, layer_st.output.dense, epsilon)
        copy_layernorm(layer_1.output.LayerNorm, layer_2.output.LayerNorm, layer_st.output.LayerNorm)

    # Pooler
    copy_linear(src1.pooler.dense, src2.pooler.dense, tgt.pooler.dense, epsilon)