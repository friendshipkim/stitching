import torch
import copy

from transformers import BertTokenizer, AutoModel, BertModel, StitchedBertConfig
from transformers.models.bert.stitch_utils import check_if_stitchable, stitch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_embeddings(x, src1, src2, tgt):
    src1_emb = src1.embeddings(x, debug=True)
    src2_emb = src2.embeddings(x, debug=True)
    tgt_emb = tgt.embeddings(x, debug=True)

    assert torch.isclose(torch.cat((src1_emb, src2_emb), dim=-1), tgt_emb).all()


def print_model_cfg(model):
    print(f'- num_parameters: {model.num_parameters()}')
    print(f"- hidden_size: {model.config.hidden_size}")
    print(f"- num_attention_heads: {model.config.num_attention_heads}")
    print(f"- num_hidden_layers: {model.config.num_hidden_layers}")
    print()


if __name__ == "__main__":
    # vocabs are identical for small and large
    tokenizer = BertTokenizer('./workspace/vocab.txt')

    # load pretrained models
    small_model = AutoModel.from_pretrained("prajjwal1/bert-mini").to(device)
    large_model = AutoModel.from_pretrained("prajjwal1/bert-small").to(device)

    # set configs to return intermediate outputs
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining.forward
    for model in [small_model, large_model]:
        model.config.output_hidden_states = True
        model.config.output_attentions = True
        model.config.test_mode = True

    # print model configs
    print("small model")
    print_model_cfg(small_model)

    print("large model")
    print_model_cfg(large_model)

    # two models to be stitched
    # TODO: replace with different bert models
    src_model_1 = small_model
    src_model_2 = copy.deepcopy(small_model)

    # stitched config / model
    stitched_config = StitchedBertConfig(**small_model.config.to_dict())
    stitched_model = BertModel(stitched_config).to(device)

    # check if the two models are stitchable
    check_if_stitchable(src_model_1.config, src_model_2.config)
    
    # stitch
    stitch(src_model_1, src_model_2, stitched_model)

    # ===== test
    # input
    text = "Example input."
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    
    # embedding
    test_embeddings(encoded_input['input_ids'], src_model_1, src_model_2, stitched_model)

    # self attention
    # bert layer
    # pooler
        
    breakpoint()
