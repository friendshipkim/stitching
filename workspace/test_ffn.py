import torch
import copy

from transformers import BertTokenizer, AutoModel, BertModel, StitchedBertConfig
from transformers.models.bert.stitch_utils import is_stitchable, stitch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    if is_stitchable(src_model_1, src_model_2):
        stitch(src_model_1, src_model_2, stitched_model)
        
    breakpoint()
