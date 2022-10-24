# development note
## proposal 1: stitch feed forward layers
### ver1 (10/24)
* stitch feed forward layers of two source BERT models with identical architectures, vocabs
  * num_hidden_layers, attention_head_size should be identical
  * hidden_size, num_attention_heads can be flexible in later versions -> change `StitchedBertConfig` class
* weights of feed forward layers are diagonally stitched, epsilon is set to 0
* Issues
  * `transformers.models.modeling_utils.apply_chunking_to_forward` function is used to acclerate high-dimensional feed forward layers after self-attention
    * used in `BertLayer` class
    * breaks the math
    * for now, just use regular linear layers instead of chunking