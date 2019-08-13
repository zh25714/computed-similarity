from tqdm import tqdm

import torch

from . import utils

def sen_pad(a):
    if a.shape[0] >= 157:
        #print("shape is big",a.shape,a)
        return a
    cat = torch.zeros(157 - a.shape[0]).type_as(a)
    a = torch.cat((a,cat),0)
    return a


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            ltree, linput, rtree, rinput, label = dataset[indices[idx]]
            dataset.num_classes = 6
            target = utils.map_label_to_target(label, dataset.num_classes)
            linput=sen_pad(linput)
            rinput=sen_pad(rinput)
            linput, rinput = linput.unsqueeze(0).to(self.device), rinput.unsqueeze(0).to(self.device)
            output = self.model(linput,rinput)
            target = target.to(self.device)
            #output = self.model(ltree, linput, rtree, rinput)
            #print(output.shape, target.shape)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            loss.backward()
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            dataset.num_classes = 6
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label = dataset[idx]
                target = utils.map_label_to_target(label, dataset.num_classes)
                target = target.to(self.device)
                linput=sen_pad(linput)
                rinput=sen_pad(rinput)
                linput, rinput = linput.unsqueeze(0).to(self.device), rinput.unsqueeze(0).to(self.device)
                output = self.model(linput,rinput)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
        #print(predictions)
        return total_loss / len(dataset), predictions
