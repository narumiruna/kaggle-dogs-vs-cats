from torch import nn


class Trainer(object):

    def __init__(self, net, optimizer, train_loader, valid_loader, device):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

    def train(self):
        cross_entropy = nn.CrossEntropyLoss()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.net(x)
            loss = cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('loss: {:.6f}'.format(loss.item()))