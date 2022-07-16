import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, output_size: int, num_layers: int = 1,
                 dropout: float = 0.2, bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden.squeeze_(0)
        output = self.fc(hidden)
        return output
