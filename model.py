import numpy as np
import torch as tc
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def train_model(train_inputs, train_labels, test_inputs, test_labels, model, lr, n_epoch=10, batch_size=64):
    optimr = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=batch_size)

    for e in range(n_epoch):
        model.train()

        for data in data_loader:
            inputs, labels = data
            optimr.zero_grad()
            outputs = model(inputs, return_series=False)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimr.step()

        model.eval()
        with tc.no_grad():
            train_preds = model(train_inputs, return_series=False)
            train_rmse = np.sqrt(loss_fn(train_preds, train_labels))
            test_preds = model(test_inputs, return_series=False)
            test_rmse = np.sqrt(loss_fn(test_preds, test_labels))
        print(f'Epoch {e+1} RMSE: (train, test) = ({train_rmse:.7f}, {test_rmse:.7f})')


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64, feature_size=5, input_window=10, label_window=5, dtype=tc.float64):
        super().__init__()
        self.lstm = None
        self.seq = nn.Sequential()
        if isinstance(hidden_size, int):
            self.lstm = nn.LSTM(feature_size, hidden_size, batch_first=True, dtype=dtype)
            self.seq.append(nn.Linear(hidden_size, label_window, dtype=dtype))
        elif isinstance(hidden_size, (list, tuple)):
            self.lstm = nn.RNN(feature_size, hidden_size[0], batch_first=True, dtype=dtype)
            for i in range(len(hidden_size)):
                if i==len(hidden_size)-1:
                    self.seq.append(nn.Linear(hidden_size[i], label_window, dtype=dtype))
                else:
                    self.seq.append(nn.Linear(hidden_size[i], hidden_size[i+1], dtype=dtype))
                    self.seq.append(nn.ReLU())

    def forward(self, input, return_series=False):
        x, h = self.lstm(input)
        if return_series:
            x = self.seq(x)
        else:
            x = self.seq(x)[:, -1, :].unsqueeze(2)
        return x