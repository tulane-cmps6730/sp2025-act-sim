import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx=0):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.slot_classifier = nn.Linear(hidden_dim * 2, num_labels)
        
    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        slot_logits = self.slot_classifier(lstm_out)
        return slot_logits

# Need to chnage this if it changes in model!
def load_lstm_model(model_path):
    VOCAB_SIZE = 526
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    NUM_LABELS = 42
    PAD_IDX = 0

    model = LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LABELS, PAD_IDX)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    return model
