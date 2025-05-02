import torch
from transformers import AutoTokenizer
from dialogue_relation_model import DialogueRelationModel

def load_minilm_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
    model_name = "microsoft/MiniLM-L12-H384-uncased"

    model = DialogueRelationModel(model_name=model_name, tokenizer=tokenizer, num_labels=4)
    state_dict = torch.load(f"{model_dir}/dialogue_relation_state_dict.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()
    return tokenizer, model
