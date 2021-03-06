import argparse
import os

import torch
from torch import nn, optim
from torchvision import models

from dataloader import get_dataloader
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--valid-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--parallel', action='store_true')
    config = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 2)

    if config.parallel and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net = net.to(device)
    optimizer = optim.Adam(net.fc.parameters(), lr=config.lr)

    train_loader, valid_loader = get_dataloader(
        config.size, config.root, config.batch_size, config.valid_ratio,
        config.num_workers)

    trainer = Trainer(net, optimizer, train_loader, valid_loader, device)
    for epoch in range(config.epochs):
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.validate()

        print('Epoch: {}/{},'.format(epoch + 1, config.epochs), \
              'train loss: {:.6f},'.format(train_loss),
              'train acc: {:.6f}.'.format(train_acc),
              'valid loss: {:.6f},'.format(valid_loss),
              'valid acc: {:.6f}.'.format(valid_acc))

        os.makedirs('models', exist_ok=True)
        torch.save(net.state_dict(),
                   'models/model_{:03d}.pth'.format(epoch + 1))


if __name__ == '__main__':
    main()
