# 1. model

import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),  # bidirectional이기 때문에 2배
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)

        z, _ = self.rnn(x)  # z = (출력+마지막상태의 은닉상태), _ = 셀상태
        # |z| = (batch_size, h, hidden_size * 2); h는 입력만큼의 결과
        z = z[:, -1]  # 이번 분류에서는 마지막 결과만 이용; 각 batch_size중에서 마지막 항목만 사용
        # |z| = (batch_size, hidden_size * 2)
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
