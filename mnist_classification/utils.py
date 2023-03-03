# 2. Data loading, preprocessor
# * MNIST
# *- MNIST는 train-set, test-set이 분리되어 있다.

import torch

from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.models.cnn_model import ConvolutionClassifier
from mnist_classification.models.rnn_model import SequenceClassifier


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.  # 0~255값을 0~1로 정규화
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)  # (N, dim); 1차원으로 변경

    return x, y


# train/valid set
def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # 1. Shuffle dataset
    # 2. split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(x,
                           dim=0,
                           index=indices
                           ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y,
                           dim=0,
                           index=indices
                           ).split([train_cnt, valid_cnt], dim=0)

    return x, y


# 학습레이어를 순열형태로 만들기
def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)  # layer당 줄어들 size

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes


def get_model(input_size, output_size, config, device):
    if config.model == 'fc':
        model = FullyConnectedClassifier(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=get_hidden_sizes(
                input_size, output_size, config.n_layers),
            use_batch_norm=not config.use_dropout,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model == 'cnn':
        model = ConvolutionClassifier(output_size)
    elif config.model == 'rnn':
        model = SequenceClassifier(
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=output_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p)
    else:
        raise NotImplementedError('You need to specify model name.')

    return model
