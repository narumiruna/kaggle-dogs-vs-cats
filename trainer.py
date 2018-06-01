import torch.nn.functional as F
from torch import nn

from utils import Accuracy, Average


class Trainer(object):

    def __init__(self, net, optimizer, train_loader, valid_loader, device):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

    def train(self):
        train_loss = Average()
        train_acc = Accuracy()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.net(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            correct = y.data.eq(output.data.argmax(dim=1)).sum()

            train_loss.update(loss.data.item(), number=x.size(0))
            train_acc.update(correct.item(), number=x.size(0))

        return train_loss.average, train_acc.accuracy
