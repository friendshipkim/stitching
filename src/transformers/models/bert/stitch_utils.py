def copy_linear(tgt, src1, src2):
    """
    tgt, src1, src2: torch.nn.Linear
    """
    # check if bias exists
    assert None not in (tgt.bias, src1.bias, src2.bias) or not any((tgt.bias, src1.bias, src2.bias))
 
    tgt_out_dim, tgt_in_dim = tgt.weight.size()
    src1_out_dim, src1_in_dim = src1.weight.size()
    src2_out_dim, src2_in_dim = src2.weight.size()

    assert tgt_out_dim == src1_out_dim + src2_out_dim
    assert tgt_in_dim == src1_in_dim + src2_in_dim

    # NOTE: check indexing
    tgt.weight.data[:src1_out_dim, :src1_in_dim] = src1.weight.data
    tgt.weight.data[-src2_out_dim:, -src2_in_dim:] = src2.weight.data
  
    if tgt.bias is not None:
        # copy bias
        tgt.bias.data[:src1_out_dim] = src1.bias.data
        tgt.bias.data[-src2_out_dim:] = src2.bias.data


def copy_layernorm(tgt, src1, src2):
    """
    tgt, src1, src2: torch.nn.modules.normalization.LayerNorm
    """
    tgt_dim, src1_dim, src2_dim = tgt.weight.size(0), src1.weight.size(0), src2.weight.size(0)
    assert tgt_dim == src1_dim + src2_dim

    # # NOTE: check indexing
    # copy weights
    tgt.weight.data[:src1_dim] = src1.weight.data
    tgt.weight.data[-src2_dim:] = src2.weight.data
