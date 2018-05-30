import argparse

from torch import nn, optim

from model import Net
from trainer import Trainer
from dataloader import get_dataloader
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--valid-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)
    config = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net(size=config.size)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)

    train_loader, valid_loader = get_dataloader(
        config.size, config.root, config.batch_size, config.valid_ratio)

    trainer = Trainer(net, optimizer, train_loader, valid_loader, device)
    for epoch in range(config.epochs):
        trainer.train()


if __name__ == '__main__':
    main()
