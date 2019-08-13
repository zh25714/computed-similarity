from copy import deepcopy

import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)-1
        y = deepcopy(labels)-1
        #print("predict",x,"ground",y)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x - y) ** 2)
