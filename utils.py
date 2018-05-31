class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count

    def __call__(self):
        return self.average


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, correct, number=1):
        self.correct += correct
        self.count += number
        self.accuracy = self.correct / self.count

    def __call__(self):
        return self.accuracy