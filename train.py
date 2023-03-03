# 4. main

# 추가할 내용
# - max_grad
# - PyTorch Ignite

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from mnist_classification.trainer import Trainer
from mnist_classification.utils import load_mnist
from mnist_classification.utils import split_data
from mnist_classification.utils import get_model


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int,
                   default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--hidden_size', type=int, default=128)  # for RNN
    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')  # 인자를 사용하는 것만으로도 true
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    p.add_argument('--model', type=str, default='fc',
                   choices=["fc", "cnn", "rnn"])

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(
        'cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=(config.model == 'fc'))
    x, y = split_data(x.to(device), y.to(device),
                      train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])  # X dim = 28 x 28 = 784
    output_size = int(max(y[0])) + 1  # 9 + 1 = 10

    model = get_model(
        input_size,
        output_size,
        config,
        device,
    )
    optimizer = optim.Adam(model.parameters())  # 최적화할 모델파라미터 설정
    crit = nn.NLLLoss()  # LogSoftmax - NLLLoss

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)
    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config,
    )

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
