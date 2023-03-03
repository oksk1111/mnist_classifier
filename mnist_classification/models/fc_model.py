# 1. model

import torch
import torch.nn as nn


# Hidden layer for Classifier
class Block(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(
                output_size) if use_batch_norm else nn.Dropout(dropout_p),
        )

    def forward(self, x):
        # n(=input)이든 m(=output)이든 k(=batch_size)가 앞에 있음을 인지하자.
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)

        return y


# 외부에 노출할 Classifier
class FullyConnectedClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):

        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(last_hidden_size, hidden_size,
                             use_batch_norm, dropout_p)]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            # Auto-grad에 의해 마지막 input이 들어가 있으며,
            # dim=-1이므로 batch_size가 아닌 input-dim이 들어가 있음
            # LogSoftmax - NLL Loss
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
