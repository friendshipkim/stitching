# development note
## proposal 1: stitch feed forward layers
### PR 1: write stitch utils
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

### PR 2: finetune stitched model on MNLI
#### List of models
1. mini
   * vanilla bert-mini
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = False 
   skip_layernorm = False 
   stitch_dummy = False   
   ```

2. small
   * vanilla bert-small 
   ```
   src_model_name = "prajjwal1/bert-small"
   model_name = "bert_small"
   do_stitch = False
   skip_layernorm = False
   stitch_dummy = False
   ```


1. mini-mini
   * stitch two identical bert-minis, eps = 0
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = True 
   skip_layernorm = False 
   stitch_dummy = False   
   ```

4. mini-mini-skipln
   * stitch two identical bert-minis except for layernorm parameters
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = True 
   skip_layernorm = True 
   stitch_dummy = False   
   ```

5. mini-random
   * stitch bert-mini and randomly initialized same sized model
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = True 
   skip_layernorm = False 
   stitch_dummy = True   
   ```

6. mini-eps
   * stitch bert-mini and the same sized model initialized with eps
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = True 
   skip_layernorm = False 
   stitch_dummy = True   
   ```
   * manually uncommented line 216-218 of `./src/transformers/models/bert/stitch_utils.py`

7. mini-mini-normaleps
   * mini-mini but eps ~ N(0, 1e-6)
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = True 
   skip_layernorm = False 
   stitch_dummy = False   
   ```
   * manually uncommented line 56-58 of `./src/transformers/models/bert/stitch_utils.py`

8. mini-mini-avgln
   * mini-mini, average layernorm params
   ```
   src_model_name = "prajjwal1/bert-mini"
   model_name = "bert_mini" 
   do_stitch = True 
   skip_layernorm = False 
   stitch_dummy = False   
   ```
   * manually changed line 84-89 of `./src/transformers/models/bert/stitch_utils.py`