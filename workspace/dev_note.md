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
* Testing stitch utils
  ```
  pytest workspace/tests/test_stitch_utils.py
  ```

#### Finetune stitched model on MNLI
List of models
1. mini
* 
`src_model_name = "prajjwal1/bert-mini"`
`model_name = "bert_mini"`
`do_stitch = False`
`skip_layernorm = False`
`stitch_dummy = False`

1. small
`src_model_name = "prajjwal1/bert-small"`
`model_name = "bert_small"`
`do_stitch = False`
`skip_layernorm = False`
`stitch_dummy = False`

3. mini-mini
`src_model_name = "prajjwal1/bert-mini"`
`model_name = "bert_mini-mini"`
`do_stitch = True`
`skip_layernorm = False`
`stitch_dummy = False`

4. mini-mini-skipln
`src_model_name = "prajjwal1/bert-mini"`
`model_name = "bert_mini-mini-skipln"`
`do_stitch = True`
`skip_layernorm = True`
`stitch_dummy = False`

5. mini-random
`src_model_name = "prajjwal1/bert-mini"`
`model_name = "bert_mini-mini-skipln"`
`do_stitch = True`
`skip_layernorm = False`
`stitch_dummy = True`

6. mini-eps
`src_model_name = "prajjwal1/bert-mini"`
`model_name = "bert_mini-mini-skipln"`
`do_stitch = True`
`skip_layernorm = False`
`stitch_dummy = True`
* manually uncommented line 216-218 of `./src/transformers/models/bert/stitch_utils.py`

7. mini-mini-normaleps
`src_model_name = "prajjwal1/bert-mini"`
`model_name = "bert_mini-mini-skipln"`
`do_stitch = True`
`skip_layernorm = False`
`stitch_dummy = True`
* manually uncommented line 56-58 of `./src/transformers/models/bert/stitch_utils.py`