# 5. prediction with test set

import argparse
import torch
import torch.nn

import sys
import numpy as np

from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.utils import load_mnist
from mnist_classification.utils import get_model


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)

    return p.parse_args().model_fn


def load(fn, device):
    d = torch.load(fn, map_location=device)

    return d['model'], d['config']


def test(model, x, y):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        # model에 input을 통과시켰을 때, 실제 정답의 개수를 단순 계산한다.
        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)


def main(model_fn):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dict, train_config = load(model_fn, device)

    # Load MNIST test set.
    x, y = load_mnist(is_train=False, flatten=(train_config.model == 'fc'))
    x, y = x.to(device), y.to(device)

    print(x.shape, y.shape)

    input_size = int(x.shape[-1])
    output_size = int(max(y)) + 1

    # model 틀을 다시 만든다
    model = get_model(
        input_size,
        output_size,
        train_config,
        device,
    )

    # model 틀에 기존 데이터값을 넣는다.
    model.load_state_dict(model_dict)

    test(model, x, y)


if __name__ == '__main__':
    model_fn = define_argparser()
    main(model_fn)
