import torch
import torch.nn as nn

class WakeWordVerifierRNN(nn.Module):
    """
    Stage 2: Heavy Temporal Verifier.
    Processes the sequence of acoustic features over time to confirm 
    the exact phonetic ordering of the wake word before final activation.
    """
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=1):
        super().__init__()
        # input_dim must match the n_mels from your DSP extractor (default is usually 40)
        
        # We utilize a Gated Recurrent Unit (GRU) to maintain sequential 
        # memory while operating with fewer parameters than an LSTM.
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # A Bidirectional GRU doubles the output context vector size
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # 1. Dimensionality Realignment
        # Input shape from our Dataset: (Batch, 1, n_mels, time_frames)
        # The RNN requires sequence format: (Batch, time_frames, n_mels)
        x = x.squeeze(1).permute(0, 2, 1)
        
        # 2. Sequential Processing
        # rnn_out contains all hidden states across time
        # hidden contains only the final hidden state of the sequence
        rnn_out, hidden = self.rnn(x)
        
        # 3. Context Extraction
        # Extract the final hidden states from both the forward and backward passes
        # hidden shape: (num_layers * num_directions, Batch, hidden_dim)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        
        # Concatenate into a single semantic context vector
        context_vector = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # 4. Final Classification
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits
        
    def count_parameters(self):
        """Diagnostic tool for memory budgeting."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)