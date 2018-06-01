class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, correct, number=1):
        self.correct += correct
        self.count += number

    @property
    def accuracy(self):
        return self.correct / self.count