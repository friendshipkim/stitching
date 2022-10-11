"""
util functions to stitch ffn layers
"""
import torch

# TODO: fill in epsilon
EPSILON = 0


# TODO: merge this into StitchedBertConfig
def is_stitchable(src1, src2):
    """
    check if given two models are stitchable
    """
    if src1.config.vocab_size != src2.config.vocab_size:
        print("vocab sizes don't match")
        return False
    if src1.config.num_hidden_layers != src2.config.num_hidden_layers:
        print("hidden layer numbers don't match")
        return False
    return True


def copy_linear(src1, src2, tgt):
    """
    src1, src2, tgt: torch.nn.Linear
    """
    # check if bias exists
    assert None not in (src1.bias, src2.bias, tgt.bias) or not any((src1.bias, src2.bias, tgt.bias))

    src1_out_dim, src1_in_dim = src1.weight.size()
    src2_out_dim, src2_in_dim = src2.weight.size()
    tgt_out_dim, tgt_in_dim = tgt.weight.size()

    # TODO: remove this
    assert tgt_out_dim == src1_out_dim + src2_out_dim
    assert tgt_in_dim == src1_in_dim + src2_in_dim

    # Initialize with epsilon
    tgt.weight.data[:] = EPSILON

    # Copy weights diagonally
    tgt.weight.data[:src1_out_dim, :src1_in_dim] = src1.weight.data
    tgt.weight.data[-src2_out_dim:, -src2_in_dim:] = src2.weight.data

    if tgt.bias is not None:
        # copy bias
        tgt.bias.data[:src1_out_dim] = src1.bias.data
        tgt.bias.data[-src2_out_dim:] = src2.bias.data


def copy_layernorm(src1, src2, tgt, debug=False):
    """
    Args:
        src1, src2, tgt (torch.nn.modules.normalization.LayerNorm)
    """
    # TODO: remove this
    src1_dim, src2_dim, tgt_dim = src1.weight.size(0), src2.weight.size(0), tgt.weight.size(0)
    assert tgt_dim == src1_dim + src2_dim

    if debug:
        # TODO: turn off layernorm
        pass
    else:
        # copy weights
        tgt.weight.data[:src1_dim] = src1.weight.data
        tgt.weight.data[-src2_dim:] = src2.weight.data

        # copy biases
        tgt.bias.data[:src1_dim] = src1.bias.data
        tgt.bias.data[-src2_dim:] = src2.bias.data


def copy_self_attn(src1, src2, tgt):
    """
    Args:
        src1, src2, tgt (transformers.models.bert.modeling_bert.BertSelfAttention)
    """
    src1_dim, src2_dim, _ = src1.query.weight.size(0), src2.query.weight.size(0), tgt.query.weight.size(0)

    for transform_type in ["query", "key", "value"]:
        # Initialize with epsilon
        tgt.get_submodule(transform_type).weight.data[:] = EPSILON

        # copy weights, src1 - top left, src2 - bottom right
        tgt.get_submodule(transform_type).weight.data[0, :src1_dim, :src1_dim] = src1.get_submodule(transform_type).weight.data
        tgt.get_submodule(transform_type).weight.data[1, -src2_dim:, -src2_dim:] = src2.get_submodule(transform_type).weight.data

        # copy biases
        # TODO: check
        tgt.get_submodule(transform_type).bias.data[0, :src1_dim] = src1.get_submodule(transform_type).bias.data
        tgt.get_submodule(transform_type).bias.data[1, -src2_dim:] = src2.get_submodule(transform_type).bias.data


def copy_embeddings(src1, src2, tgt):
    """
    Args:
        src1, src2, tgt (transformers.models.bert.modeling_bert.BertEmbeddings)
    """
    # embeddings
    embed_types = ["word_embeddings", "position_embeddings", "token_type_embeddings"]
    for embed_type in embed_types:
        tgt.get_submodule(embed_type).weight.data[:] = torch.cat((
            src1.get_submodule(embed_type).weight.data,
            src2.get_submodule(embed_type).weight.data,
        ), dim=-1)

    # layernorm
    copy_layernorm(src1.LayerNorm, src2.LayerNorm, tgt.LayerNorm)


def copy_attentions(src1, src2, tgt):
    """
    Args:
        src1, src2, tgt (transformers.models.bert.modeling_bert.BertAttention)
    """

    # key, query, value transformations
    copy_self_attn(src1.self, src2.self, tgt.self)

    # linear
    copy_linear(src1.output.dense, src2.output.dense, tgt.output.dense)

    # layernorm
    copy_layernorm(src1.output.LayerNorm, src2.output.LayerNorm, tgt.output.LayerNorm)


def stitch(src1, src2, tgt):
    """
    Args:
        src1, src2 (transformer.BertModel): small bert models to stitch
    """
    # embeddings
    copy_embeddings(src1.embeddings, src2.embeddings, tgt.embeddings)

    # copy transformer layers
    for layer_1, layer_2, layer_st in zip(src1.encoder.layer, src2.encoder.layer, tgt.encoder.layer):
        # multihead attentions
        copy_attentions(layer_1.attention, layer_2.attention, layer_st.attention)

        # intermediate ffn
        copy_linear(layer_1.intermediate.dense, layer_2.intermediate.dense, layer_st.intermediate.dense)

        # output ffn
        copy_linear(layer_1.output.dense, layer_2.output.dense, layer_st.output.dense)
        copy_layernorm(layer_1.output.LayerNorm, layer_2.output.LayerNorm, layer_st.output.LayerNorm)

    # pooler
    copy_linear(src1.pooler.dense, src2.pooler.dense, tgt.pooler.dense)
