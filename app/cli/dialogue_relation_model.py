import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DialogueRelationModel(nn.Module):
    def __init__(self, model_name, tokenizer, num_labels):
        super().__init__()
        self.config  = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.dropout    = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_rep)
        logits = self.classifier(x)
        return logits
