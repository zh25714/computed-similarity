import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import map_label_to_target
from . import Constants
from abcnn import Abcnn3


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), F.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        print("是否启动")
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        print("vec_dist",vec_dist.shape)
        print("wh(vec_dist)",self.wh(vec_dist).shape)

        out = torch.sigmoid(self.wh(vec_dist))
        print("out = torch.sigmoid(self.wh(vec_dist))",out.shape)

        print("wp(out)",self.wp(out).shape)
        out = F.log_softmax(self.wp(out), dim=1)
        print("out = F.log_softmax(self.wp(out), dim=1)",out.shape)
        return out
        
'''
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        # sparsity = True

        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = nn.CosineSimilarity(dim=1,eps=1e-9)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        output = map_label_to_target(output,6)
        return output
'''

# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        #sparsity = True
       
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output


class ABCNN(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(ABCNN, self).__init__()
        # sparsity = True
        self.emb_dim = in_dim
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.emb.weight.requires_grad = False
        #self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.abcnn = Abcnn3(emb_dim=157, sentence_length=300,filter_width=3)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
        #size (batch_size, 1, sentence_length, emb_dim)
    def forward(self, linputs,  rinputs):
        #print(linputs.shape)
        linputs = self.emb(linputs).reshape((linputs.shape[0],1,linputs.shape[1],self.emb_dim)).transpose(-1,-2)
        rinputs = self.emb(rinputs).reshape((rinputs.shape[0],1,rinputs.shape[1],self.emb_dim)).transpose(-1,-2)
        output,r1,r2 = self.abcnn(linputs,rinputs)
        #print(output)
        #print("repre",r1.shape,r2.shape)
        return self.similarity(r1,r2)

class Hybrid(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(Hybrid, self).__init__()
        # sparsity = True
        self.emb_dim = in_dim
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.abcnn = Abcnn3(emb_dim=300, sentence_length=157,filter_width=4)
        self.similarity = Similarity(2*mem_dim, hidden_dim, num_classes)
        #size (batch_size, 1, sentence_length, emb_dim)
    def forward(self, ltree, linputs, rtree, rinputs, linput_pad, rinput_pad):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        #print(linputs.shape)
        linputs = self.emb(linput_pad).reshape((linput_pad.shape[0],1,linput_pad.shape[1],self.emb_dim))
        rinputs = self.emb(rinput_pad).reshape((rinput_pad.shape[0],1,rinput_pad.shape[1],self.emb_dim))
        output,r1,r2 = self.abcnn(linputs,rinputs)
        
        repre1 = torch.cat((r1,lstate),1)
        repre2 = torch.cat((r2,rstate),1)
        '''
        repre1 = r1+lstate
        repre2 = r2+rstate
        '''
        return self.similarity(repre1,repre2)
