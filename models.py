import os

import numpy as np
import torch
import torch.nn as nn
from torchtext.legacy import data


class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: float = 0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class LSTM(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, output_size: int, text_field: data.Field,
                 num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.text_field = text_field
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

    def train_model(self, train_dataloader: data.BucketIterator, valid_dataloader: data.BucketIterator,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: int = 50,
                    early_stop: bool = False):
        if early_stop:
            early_stopping = EarlyStopping(min_delta=0.02)
        total_train_loss, total_valid_loss = [], []
        total_train_accuracy, total_valid_accuracy = [], []

        for epoch in np.arange(1, epochs + 1):
            print(f'Epoch {epoch}:')
            self.train()
            train_losses, valid_loss = [], []
            for batch_idx, batch in enumerate(train_dataloader):
                X = batch.text.to(device)
                t = batch.airline_sentiment.to(device)

                y = self.forward(X)
                loss = criterion(y, t)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_accuracy = LSTM.get_accuracy(self, train_dataloader, device)

            print(f'train_loss: {np.mean(train_losses)}, train_accuracy: {train_accuracy}.')
            total_train_loss.append(np.mean(train_losses))
            total_train_accuracy.append(train_accuracy)

            self.eval()
            with torch.no_grad():
                for batch in valid_dataloader:
                    X = batch.text.to(device)
                    t = batch.airline_sentiment.to(device)
                    y = self.forward(X)
                    loss = criterion(y, t)
                    valid_loss.append(loss.item())

                valid_accuracy = LSTM.get_accuracy(self, valid_dataloader, device)
                print(f'valid_loss: {np.mean(valid_loss)}, valid_accuracy: {valid_accuracy}.')

                total_valid_loss.append(np.mean(valid_loss))
                total_valid_accuracy.append(valid_accuracy)

            if early_stop:
                early_stopping(np.mean(train_losses), np.mean(valid_loss))
                if early_stopping.early_stop:
                    print('Early stopping!')
                    break

        return total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy

    def predict(self, text: str, device: torch.device) -> tuple[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            token_indices = [self.text_field.vocab.stoi[token] for token in self.text_field.tokenize(text)]
            token_tensor = torch.LongTensor(token_indices).to(device).unsqueeze(1)
            prediction = torch.nn.functional.softmax(self.forward(token_tensor), dim=1)
            return 'negative' if prediction.argmax().item() == 0 else 'positive', prediction

    def save(self, path: str):
        torch.save(self, os.getcwd() + f'/{path}/model.pt')

    @staticmethod
    def load(path: str):
        model = torch.load(os.path.join(os.getcwd(), path, 'model.pt'))
        return model

    @staticmethod
    def get_accuracy(model: nn.Module, dataloader: data.BucketIterator, device: torch.device) -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                X = batch.text.to(device)
                t = batch.airline_sentiment.to(device)
                y = model.forward(X)
                _, y_pred = torch.max(y, 1)
                correct += (y_pred == t).sum().item()
        return correct / len(dataloader.dataset)
