import torch
import torch.nn as nn
import math
from ..module.gather import GatherLayer
import torch.distributed as dist
from util.tensor_op import cos_sim

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.worldsize = dist.get_world_size()
        self.batch_size = batch_size * self.worldsize
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)


        self.mask = self.mask_correlated_samples(self.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def mask_correlated_samples(self, batch_size):

        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        # z_i = GatherLayer.apply(z_i)
        # z_j = GatherLayer.apply(z_j)
        # z_i = torch.cat(z_i, dim=0)
        # z_j = torch.cat(z_j, dim=0)

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # sim = torch.matmul(z, z.T) / self.temperature
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # print("argmax")
        # print(torch.argmax(sim1,dim=1),torch.argmax(sim,dim=1))
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        '''contrastive.py 的 dist_infonce'''
        # scores = cos_sim(z_i, z_j) / self.temperature
        # # print(scores)
        # batch_size = len(z_i)
        # # print(batch_size)
        # # print(torch.distributed.get_rank())
        # _labels = torch.tensor(range(batch_size), dtype=torch.long, device=scores.device)
        # # print(_labels)
        # # raise ValueError("ddddddddd")
        #
        # print("cc infonce loss")
        # print(scores)
        # print(torch.argmax(scores, dim=1))
        # cross_entropy_loss = nn.CrossEntropyLoss()
        # loss = cross_entropy_loss(scores, _labels)
        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature


        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        print("mask:{}".format(mask))
        return mask

    def forward(self, c_i, c_j):
        # c_i = GatherLayer.apply(c_i)
        # c_j = GatherLayer.apply(c_j)
        # c_i = torch.cat(c_i, dim=0)
        # c_j = torch.cat(c_j, dim=0)
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        # print("sim,labels")
        # print(sim,labels,logits)
        return loss + ne_loss
