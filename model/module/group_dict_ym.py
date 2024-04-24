import torch
import torch.nn as nn
import torch.distributed as dist
from util.tensor_op import concat_all_gather, GeLU
import numpy as np


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))


def ema_tensor_inplace(moving_avg, new, decay):
    new_out = torch.mul(new, 1.0 - decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())


# point-to-point updates the group where user participate
def ema_tensor_mask_dict_inplace(moving_avg, new, mask, decay):
    new_out = torch.mul(new, 1 - decay)
    moving_avg[mask==1] = moving_avg.data[mask==1] * decay
    moving_avg.add_(new_out.detach())


# Each group has its own decay_rate
def ema_tensor_mask_group_inplace(moving_avg, new, mask, decay, cluster_update_num):
    for idx, m in enumerate(mask):
        if m == 0:
            continue
        # if cluster_update_num[idx] <= 10:
        #     decay = decay * cluster_update_num[idx] / 10.
        new_out = new[idx] * (1-decay)
        moving_avg[idx] = moving_avg.data[idx] * decay
        moving_avg[idx] = moving_avg.data[idx] + new_out.detach()


def sum_inplace(sum_data, new):
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


# def laplace_smoothing_dim(x, n_categories,dim=1, eps=1e-5):
#     return (x + eps) / (x.sum(dim=dim, keepdim=True) + n_categories * eps)


class InterestDict(nn.Module):
    def __init__(self, num_interest, dim_interest, decay=0.05, max_decay=0.95, eps=1e-5, topK=1):
        super(InterestDict, self).__init__()
        self._num_interest = num_interest
        self._dim_interest = dim_interest

        dictionary = torch.randn(num_interest, dim_interest,requires_grad=False)
        self.register_buffer('dictionary',dictionary)
        # nn.init.normal_(self.dictionary)
        self.register_buffer('cluster_size', torch.zeros(num_interest).unsqueeze(0))

        self.register_buffer('cluster_size_sum', torch.zeros(num_interest))
        # self.register_buffer('cluster_value', torch.zeros(num_interest, dim_interest))
        self.register_buffer('num_update', torch.tensor(1))
        self.register_buffer('cluster_update_num', torch.ones(num_interest)) # group updated counts

        self.decay = decay
        self.eps = eps
        self.curr_decay = self.decay
        self.max_decay = max_decay
        self.K = topK
        self.init_dict_num = 0

    def set_decay_updates(self, num_update):
        self.curr_decay = min(0.01*(num_update/10 +1), self.max_decay)
        if num_update < 1000:
            self.num_update += 1




    def _init_dict_by_StdMean(self, input_flatten):
        # x_std,x_mean = torch.std_mean(torch.mean(input_flatten,dim=0))
        # nn.init.normal_(self.dictionary.detach(),std=float(x_std),mean=float(x_mean))

        # x_min, x_max = torch.min(input_flatten), torch.max(input_flatten)
        # nn.init.uniform_(self.dictionary.detach(), a=-1, b=1)
        # self.cluster_value.data.copy_(self.dictionary.clone())

        input_flatten_all = concat_all_gather(input_flatten.clone())
        input_len = input_flatten_all.shape[0]
        init_embed = []
        for i in range(self._num_interest):
            proto_mean = torch.mean(input_flatten_all[list(np.random.randint(0, input_len, size=1)), :],dim=0)
            x_std, x_mean = torch.std_mean(proto_mean, dim=-1)
            nn.init.normal_(proto_mean, std=float(x_std), mean=float(x_mean))
            init_embed.append(proto_mean)
        init_embed = torch.stack(init_embed, dim=0)
        torch.distributed.broadcast(init_embed, src=0)
        self.dictionary.data = init_embed


    def _init_dict_by_user(self, input_flatten, idx_start):
        # input_flatten_all = concat_all_gather(input_flatten.clone())
        input_flatten_all = input_flatten
        input_len = input_flatten_all.shape[0]
        init_embed = [
            torch.mean(
                input_flatten_all[list(np.random.randint(0, input_len, size=2)), :],
                dim=0)
            for i in range(self._num_interest)
        ]
        init_embed = torch.stack(init_embed, dim=0)

        # self.init_dict_num += input_len
        #
        # if self._num_interest / self.init_dict_num > 1:
        #     self.dictionary.data[idx_start:idx_start + input_len, :] = input_flatten.clone()
        # else:
        #     if idx_start == 0:
        #         self.dictionary.data[idx_start:, :] = input_flatten[:self._num_interest, :].clone()
        #     else:
        #         self.dictionary.data[idx_start:, :] = input_flatten[:self._num_interest % idx_start, :].clone()

        # torch.distributed.broadcast(init_embed, src=0)
        self.dictionary.data = init_embed
        # self.cluster_value.data.copy_(self.dictionary.clone())

    def _get_topK_emb(self, distances):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=False, dim=-1)
        topk_idx = idx[:, :self.K]
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        group_emb = topk_emb_sum.clone() / self.K
        return group_emb, topk_idx

    def forward(self, inputs_flatten):
        # random initial
        # if self.num_update == 1:
        #     # self._init_dict_by_StdMean(inputs_flatten)
        #     self._init_dict_by_user(inputs_flatten, 0)
        #     torch.distributed.barrier()

        print(inputs_flatten)
        print(self.dictionary)

        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.dictionary.detach() ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))
        # distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        group_emb, topk_idx = self._get_topK_emb(distances)

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encoding_indices = topk_idx
        # print(encoding_indices)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                device=inputs_flatten.device)
        encodings.scatter_(1, encoding_indices, 1)  # one-hot
        # print(encodings)

        if self.training:
            self.set_decay_updates(self.num_update)

            # The following code is used for updating 'dictionary' buffer
            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            self.cluster_size = torch.sum(concat_all_gather(tmp_sum), dim=0) # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size!=0] = 1

            sum_inplace(self.cluster_size_sum, self.cluster_size)
            sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            input_sum = torch.sum(concat_all_gather(input_sum_tmp.unsqueeze(dim=0)), dim=0) # cluster_num * dim

            world_size = dist.get_world_size()
            input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)
            # ema_tensor_mask_dict_inplace(self.cluster_value, input_sum, cluster_mask, self.curr_decay)
            ema_tensor_mask_group_inplace(self.dictionary, input_sum, cluster_mask, self.max_decay,
                                          self.cluster_update_num)

            # print(self.cluster_value)
            # dist.all_reduce(self.cluster_value.div_(world_size))
            # print(self.cluster_value)
            # self.dictionary.data.copy_(self.cluster_value)

        print("update dict")
        print(self.cluster_size, self.cluster_size_sum, self.cluster_update_num)

        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten
        return group_emb, topk_idx

class InterestDictSoft(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.05, max_decay=0.95, eps=1e-5, topK=1):
        super(InterestDictSoft, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)

    def forward(self, inputs_flatten):
        # random initial
        if self.num_update == 1:
            # self._init_dict_by_StdMean(inputs_flatten)
            self._init_dict_by_user(inputs_flatten, 0)
            torch.distributed.barrier()

        print(inputs_flatten)
        print(self.dictionary)

        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.dictionary.detach() ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))
        # distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        print("distances")
        print(distances)
        group_emb, topk_idx = self._get_topK_emb(distances)

        if self.training:
            self.set_decay_updates(self.num_update)

            # distances = nn.functional.normalize(distances, p=2, dim=1)
            scale_min = torch.min(distances)
            scale_max = torch.max(distances)
            distances = (distances - scale_min) / (scale_max - scale_min)
            print("norm distances")
            print(distances)

            encodings = torch.softmax(distances * -20, dim=1)
            print(encodings)
            print(torch.max(encodings, dim=0))
            # print(torch.nonzero(encodings))

            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            self.cluster_size = torch.sum(concat_all_gather(tmp_sum), dim=0)  # 1*cluster_num

            sum_inplace(self.cluster_size_sum, self.cluster_size)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            input_sum = torch.sum(concat_all_gather(input_sum_tmp.unsqueeze(dim=0)), dim=0)  # cluster_num * dim
            print(input_sum.shape)
            input_sum = input_sum / self.cluster_size.unsqueeze(1)

            ema_tensor_inplace(self.dictionary, input_sum, self.max_decay)

        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten
        print("update dict")
        print(self.cluster_size, self.cluster_size_sum)

        return group_emb, topk_idx, distances

class InterestDictSoft2(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.05, max_decay=0.95, eps=1e-5, topK=1):
        super(InterestDictSoft2, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)

    def forward(self, inputs_flatten):
        # random initial
        if self.num_update == 1:
            # self._init_dict_by_StdMean(inputs_flatten)
            self._init_dict_by_user(inputs_flatten, 0)
            torch.distributed.barrier()

        print(inputs_flatten)
        print(self.dictionary)

        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.dictionary.detach() ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))
        # distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        print(distances)
        group_emb, topk_idx = self._get_topK_emb(distances)

        if self.training:
            self.set_decay_updates(self.num_update)

            # distances = nn.functional.normalize(distances, p=2, dim=1)
            scale_min = torch.min(distances)
            scale_max = torch.max(distances)
            distances = (distances - scale_min) / (scale_max - scale_min)
            print(distances)

            encodings = torch.softmax(distances * -20, dim=1)
            print(encodings)
            print(torch.max(encodings, dim=0))
            # print(torch.nonzero(encodings))

            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            self.cluster_size = torch.sum(concat_all_gather(tmp_sum), dim=0)  # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            sum_inplace(self.cluster_size_sum, self.cluster_size)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            input_sum = torch.sum(concat_all_gather(input_sum_tmp.unsqueeze(dim=0)), dim=0)  # cluster_num * dim
            print(input_sum)
            input_sum[cluster_mask == 1] = input_sum[cluster_mask == 1] / self.cluster_size[
                cluster_mask == 1].unsqueeze(1)

            ema_tensor_mask_group_inplace(self.dictionary, input_sum, cluster_mask, self.max_decay,
                                          self.cluster_update_num)

        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten
        print("update dict")
        print(self.cluster_size, self.cluster_size_sum)

        return group_emb, topk_idx

class InterestDictSoft_uni(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDictSoft_uni, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)
        self.register_buffer('topk_cluster_size_sum', torch.zeros(num_interest))
        self.dict_tmp = self.dictionary.clone().detach()

    def reset_cluster_size(self):
        self.cluster_size_sum.data = torch.zeros_like(self.cluster_size_sum)
    def set_decay(self,epoch, curr_step=0, total_step=1000):
        decay = 0.8 + epoch * 0.05
        self.curr_decay = min(decay, self.max_decay)
        print("set coarse decay:{}".format(self.curr_decay))

    def set_dict_init(self,proto_dict):
        print("set dict init")
        print(proto_dict)
        print(self.dictionary)
        self.dictionary.data = proto_dict
        self.dict_tmp = self.dictionary.clone().detach()
        print("coarse after")
        print(self.dictionary,self.dictionary.shape)

    def _get_topK_emb(self, distances, K=1):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=True, dim=-1)
        topk_idx = idx[:, :K]
        distance = distence[:, :K]
        distance_neg = distence[:, -K:]
        print("topk coarse dist")
        print(distance, distance_neg)
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        # topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        # group_emb = topk_emb_sum / self.K

        topk_w = torch.softmax(distance,dim=1)
        group_emb = torch.matmul(topk_w.unsqueeze(1),topk_emb)

        group_emb = group_emb.reshape([distances.shape[0],-1])
        return group_emb, topk_idx[:,:5], distance[:,:5], topk_emb[:,0]

    def forward(self, inputs_flatten):
        # random initial
        # if self.num_update == 1:
        #     # self._init_dict_by_StdMean(inputs_flatten)
        #     self._init_dict_by_user(inputs_flatten, 0)
        #     torch.distributed.barrier()

        print("input,coarse dict")
        print(self.curr_decay)
        print(inputs_flatten)
        print(self.dictionary)

        # euc_distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        #              + torch.sum(self.dictionary.detach() ** 2, dim=1)
        #              - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))

        # distances = nn.functional.normalize(euc_distances, p=2, dim=1)
        # print("distances")
        # print(distances,euc_distances)
        # print(torch.max(distances, dim=1))


        # his_distances = nn.CosineSimilarity(dim=2)(his_embedding.unsqueeze(1), self.dictionary.unsqueeze(0))
        # his_group_emb, his_topk_idx, his_topk_dis, his_topK_emb = self._get_topK_emb(his_distances, self.K)
        distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        group_emb, topk_idx, topk_dis, top1_emb = self._get_topK_emb(distances, self.K)

        if self.training:
            # self.set_decay_updates(self.num_update)
            topk_update = 0
            if topk_update:
                tmp_dist = distances.clone()
                tmp_dist[tmp_dist <= 0] = -1e5
                encodings_W = torch.softmax(tmp_dist * 20, dim=1)
                print("encoding_W:{},{}".format(encodings_W, tmp_dist))
                topk_encodings = encodings_W
            elif True:
                encodings_W = torch.softmax(topk_dis * 20, dim=1)
                scale_W = topk_dis[:, 0] / encodings_W[:, 0]
                encodings_W = encodings_W * scale_W.unsqueeze(-1)
                # encodings_W = torch.softmax(topk_dis.flatten(), dim=-1).reshape([topk_dis.shape[0],-1]) * topk_dis.shape[1]
                encoding_indices = topk_idx
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)
            else:
                # encodings_W = torch.softmax(topk_dis * 20, dim=1)
                # encodings_W = encodings_W[:, :1]

                encodings_W = topk_dis[:, :1]
                encoding_indices = topk_idx[:, :1]
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot

            # encodings = torch.softmax(distances * 20, dim=1)
            encodings = topk_encodings
            # print(encodings)
            # print(torch.max(encodings, dim=1))

            self.cluster_size = torch.sum(encodings, dim=0, keepdim=True)
            # print(torch.max(self.cluster_size, dim=1))
            # self.cluster_size = concat_all_gather(tmp_sum)  # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            cluster_size_all = torch.sum((self.cluster_size),dim=0)
            sum_inplace(self.cluster_size_sum, cluster_size_all)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum = torch.matmul(encodings.t(), inputs_flatten).unsqueeze(0)


            input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)

            # input_sum = input_sum / self.cluster_size.unsqueeze(1)
            # print(input_sum,input_sum.shape)
            # print("dict")
            # print(self.dictionary)
            input_sum_all = (input_sum)  # w * cluster_num * dim
            cluster_mask_all = (cluster_mask)
            for input_sum_one,cluster_mask_one in zip(input_sum_all,cluster_mask_all):
                # ema_tensor_inplace(self.dictionary, input_sum_one, self.curr_decay)
                ema_tensor_mask_group_inplace(self.dictionary, input_sum_one, cluster_mask_one, self.curr_decay,self.cluster_update_num)
                # print("dist dict:{},{}".format(input_sum_one,cluster_mask_one))
            # print("dict updated:{}".format(self.dictionary))
            print("update coarse dict")
            print(self.cluster_size, self.cluster_size_sum, topk_idx[:, :10])

        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten
        return group_emb, topk_idx, distances
        # group_emb, topk_idx, topk_dis, top1_emb = self._get_topK_emb(distances)
        # his_group_emb = (his_group_emb - his_embedding).detach() + his_embedding
        # return his_group_emb, his_topk_idx, his_distanceK_embs, his_top

    def _get_dictionary(self):
        return self.dictionary.data


class InterestDictSoft_cosv2(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDictSoft_cosv2, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)
        self.register_buffer('topk_cluster_size_sum', torch.zeros(num_interest))
        self.dict_tmp = self.dictionary.clone().detach()

    def set_decay(self,epoch, curr_step=0, total_step=1000):
        decay = 0.8 + epoch * 0.05
        self.curr_decay = min(decay, self.max_decay)
        print("set decay:{}".format(self.curr_decay))

    def reset_cluster_size(self):
        self.cluster_size_sum.data = torch.zeros_like(self.cluster_size_sum)

    def set_dict_init(self,proto_dict):
        print("set dict init")
        print(proto_dict)
        print(self.dictionary)
        self.dictionary.data = proto_dict
        self.dict_tmp = self.dictionary.clone().detach()
        print("after")
        print(self.dictionary,self.dictionary.shape)

    def _get_topK_emb(self, distances,K):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distances_s, idx = torch.sort(distances, descending=True, dim=-1)
        topk_idx = idx[:, :K]
        distance = distances_s[:, :K]
        distance_neg = distances_s[:, -K:]
        print("topk dist")
        print(distance, distance_neg)
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        # topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        # group_emb = topk_emb_sum / self.K

        topk_w = torch.softmax(distance,dim=1)
        group_emb = torch.matmul(topk_w.unsqueeze(1),topk_emb)

        group_emb = group_emb.reshape([distances.shape[0],-1])
        return group_emb, topk_idx[:,:5], distance[:,:5], topk_emb[:0]

    def forward(self, inputs_flatten, his_embedding):
        # random initial
        # if self.num_update == 1:
        #     # self._init_dict_by_StdMean(inputs_flatten)
        #     self._init_dict_by_user(inputs_flatten, 0)
        #     torch.distributed.barrier()

        # print("input,dict")
        #
        # print(inputs_flatten)
        # print(self.dictionary)

        # euc_distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        #              + torch.sum(self.dictionary.detach() ** 2, dim=1)
        #              - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))

        his_distances  = nn.CosineSimilarity(dim=2)(his_embedding.unsqueeze(1), self.dictionary.unsqueeze(0))
        his_group_emb, his_topk_idx, his_topk_dis, his_topK_emb = self._get_topK_emb(his_distances, self.K)


        # print("distances")
        # print(distances,euc_distances)
        # print(torch.max(distances, dim=1))


        if self.training:
            # self.set_decay_updates(self.num_update)

            distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
            group_emb, topk_idx, topk_dis, topK_emb = self._get_topK_emb(distances, self.K)

            topk_update = 0
            if topk_update:
                tmp_dist = distances.clone()
                tmp_dist[tmp_dist <= 0] = -1e5
                encodings_W = torch.softmax(tmp_dist * 20, dim=1)
                print("encoding_W:{},{}".format(encodings_W, tmp_dist))
                topk_encodings = encodings_W
            elif True:
                encodings_W = torch.softmax(topk_dis * 20, dim=1)
                scale_W = topk_dis[:,0]/encodings_W[:,0]
                encodings_W = encodings_W * scale_W.unsqueeze(-1)
                # encodings_W = torch.softmax(topk_dis.flatten(), dim=-1).reshape([topk_dis.shape[0],-1]) * topk_dis.shape[1]
                encoding_indices = topk_idx
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot
            else:
                # encodings_W = torch.softmax(topk_dis * 20, dim=1)
                # encodings_W = encodings_W[:, :1]

                encodings_W = topk_dis[:, :1]
                encoding_indices = topk_idx[:, :1]
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot

            # encodings = torch.softmax(distances * 20, dim=1)
            encodings = topk_encodings
            print(encodings)
            print(torch.max(encodings, dim=1))

            self.cluster_size = torch.sum(encodings, dim=0, keepdim=True)
            print(torch.max(self.cluster_size, dim=1))
            # self.cluster_size = concat_all_gather(tmp_sum)  # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            cluster_size_all = torch.sum(concat_all_gather(self.cluster_size),dim=0)
            sum_inplace(self.cluster_size_sum, cluster_size_all)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum = torch.matmul(encodings.t(), inputs_flatten).unsqueeze(0)


            input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)

            # input_sum = input_sum / self.cluster_size.unsqueeze(1)

            input_sum_all = concat_all_gather(input_sum)  # w * cluster_num * dim
            cluster_mask_all = concat_all_gather(cluster_mask)
            for input_sum_one,cluster_mask_one in zip(input_sum_all,cluster_mask_all):
                # ema_tensor_inplace(self.dictionary, input_sum_one, self.curr_decay)
                ema_tensor_mask_group_inplace(self.dictionary, input_sum_one, cluster_mask_one, self.curr_decay,self.cluster_update_num)
                # print("dist dict:{},{}".format(input_sum_one,cluster_mask_one))
            # print("dict updated:{}".format(self.dictionary))
            print("update dict")
            print(self.cluster_size, self.cluster_size_sum, topk_idx[:, :10])

            # # topk update
            # encodings_weight = torch.softmax(topk_dis * 20, dim=1)
            # encoding_indices = topk_idx
            # topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
            #                              device=inputs_flatten.device)
            # topk_encodings.scatter_(1, encoding_indices, encodings_weight)  # one-hot
            # print("topk encoding")
            # print(torch.max(topk_encodings, dim=1))
            #
            # topk_tmp_sum = torch.sum(topk_encodings, dim=0, keepdim=True)
            # print(torch.max(topk_tmp_sum, dim=1))
            # self.topk_cluster_size = torch.sum(concat_all_gather(topk_tmp_sum), dim=0)  # 1*cluster_num
            #
            # topk_cluster_mask = torch.zeros_like(self.topk_cluster_size)
            # topk_cluster_mask[self.topk_cluster_size != 0] = 1
            #
            # sum_inplace(self.topk_cluster_size_sum, self.topk_cluster_size)
            # # sum_inplace(self.cluster_update_num, cluster_mask)
            #
            # topk_input_sum_tmp = torch.matmul(topk_encodings.t(), inputs_flatten)
            # topk_input_sum = torch.sum(concat_all_gather(topk_input_sum_tmp.unsqueeze(dim=0)),
            #                            dim=0)  # cluster_num * dim
            #
            # topk_input_sum[topk_cluster_mask == 1] = topk_input_sum[topk_cluster_mask == 1] / self.topk_cluster_size[
            #     topk_cluster_mask == 1].unsqueeze(1)
            #
            # ema_tensor_mask_group_inplace(self.dictionary, input_sum, topk_cluster_mask, self.max_decay,
            #                               self.cluster_update_num)
            # print("topk update dict")
            # print(self.topk_cluster_size, self.topk_cluster_size_sum)

        # group_emb, topk_idx, topk_dis, top1_emb = self._get_topK_emb(distances)
        # group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten
        his_group_emb = (his_group_emb - his_embedding).detach() + his_embedding


        return his_group_emb, his_topk_idx, his_distances, his_topK_emb

    def _get_dictionary(self):
        return self.dictionary.data



class InterestDictSoft_coarse(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDictSoft_coarse, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)
        self.register_buffer('topk_cluster_size_sum', torch.zeros(num_interest))
        self.dict_tmp = self.dictionary.clone().detach()

    def reset_cluster_size(self):
        self.cluster_size_sum.data = torch.zeros_like(self.cluster_size_sum)
    def set_decay(self,epoch, curr_step=0, total_step=1000):
        decay = 0.8 + epoch * 0.1 + 0.1/total_step * curr_step
        self.curr_decay = min(decay, self.max_decay)
        print("set coarse decay:{}".format(self.curr_decay))

    def set_dict_init(self,proto_dict):
        print("set dict init")
        print(proto_dict)
        print(self.dictionary)
        self.dictionary.data = proto_dict
        self.dict_tmp = self.dictionary.clone().detach()
        print("coarse after")
        print(self.dictionary,self.dictionary.shape)

    def _get_topK_emb(self, distances, K=1):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=True, dim=-1)
        topk_idx = idx[:, :K]
        distance = distence[:, :K]
        distance_neg = distence[:, -K:]
        print("topk coarse dist")
        print(distance, distance_neg)
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        # topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        # group_emb = topk_emb_sum / self.K

        topk_w = torch.softmax(distance,dim=1)
        group_emb = torch.matmul(topk_w.unsqueeze(1),topk_emb)

        group_emb = group_emb.reshape([distances.shape[0],-1])
        return group_emb, topk_idx[:,:5], distance[:,:5], topk_emb[:,0]

    def forward(self, inputs_flatten, his_embedding):
        # random initial
        # if self.num_update == 1:
        #     # self._init_dict_by_StdMean(inputs_flatten)
        #     self._init_dict_by_user(inputs_flatten, 0)
        #     torch.distributed.barrier()

        print("input,coarse dict")
        print(self.curr_decay)
        print(inputs_flatten)
        print(self.dictionary)

        # euc_distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        #              + torch.sum(self.dictionary.detach() ** 2, dim=1)
        #              - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))

        # distances = nn.functional.normalize(euc_distances, p=2, dim=1)
        # print("distances")
        # print(distances,euc_distances)
        # print(torch.max(distances, dim=1))


        his_distances = nn.CosineSimilarity(dim=2)(his_embedding.unsqueeze(1), self.dictionary.unsqueeze(0))
        his_group_emb, his_topk_idx, his_topk_dis, his_topK_emb = self._get_topK_emb(his_distances, self.K)

        if self.training:
            # self.set_decay_updates(self.num_update)

            distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
            group_emb, topk_idx, topk_dis, top1_emb = self._get_topK_emb(distances, self.K)

            topk_update = 0
            if topk_update:
                tmp_dist = distances.clone()
                tmp_dist[tmp_dist <= 0] = -1e5
                encodings_W = torch.softmax(tmp_dist * 20, dim=1)
                print("encoding_W:{},{}".format(encodings_W, tmp_dist))
                topk_encodings = encodings_W
            elif True:
                encodings_W = torch.softmax(topk_dis * 20, dim=1)
                scale_W = topk_dis[:, 0] / encodings_W[:, 0]
                encodings_W = encodings_W * scale_W.unsqueeze(-1)
                # encodings_W = torch.softmax(topk_dis.flatten(), dim=-1).reshape([topk_dis.shape[0],-1]) * topk_dis.shape[1]
                encoding_indices = topk_idx
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)
            else:
                # encodings_W = torch.softmax(topk_dis * 20, dim=1)
                # encodings_W = encodings_W[:, :1]

                encodings_W = topk_dis[:, :1]
                encoding_indices = topk_idx[:, :1]
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot

            # encodings = torch.softmax(distances * 20, dim=1)
            encodings = topk_encodings
            # print(encodings)
            # print(torch.max(encodings, dim=1))

            self.cluster_size = torch.sum(encodings, dim=0, keepdim=True)
            # print(torch.max(self.cluster_size, dim=1))
            # self.cluster_size = concat_all_gather(tmp_sum)  # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            cluster_size_all = torch.sum(concat_all_gather(self.cluster_size),dim=0)
            sum_inplace(self.cluster_size_sum, cluster_size_all)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum = torch.matmul(encodings.t(), inputs_flatten).unsqueeze(0)


            input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)

            # input_sum = input_sum / self.cluster_size.unsqueeze(1)
            # print(input_sum,input_sum.shape)
            # print("dict")
            # print(self.dictionary)
            input_sum_all = concat_all_gather(input_sum)  # w * cluster_num * dim
            cluster_mask_all = concat_all_gather(cluster_mask)
            for input_sum_one,cluster_mask_one in zip(input_sum_all,cluster_mask_all):
                # ema_tensor_inplace(self.dictionary, input_sum_one, self.curr_decay)
                ema_tensor_mask_group_inplace(self.dictionary, input_sum_one, cluster_mask_one, self.curr_decay,self.cluster_update_num)
                # print("dist dict:{},{}".format(input_sum_one,cluster_mask_one))
            # print("dict updated:{}".format(self.dictionary))
            print("update coarse dict")
            print(self.cluster_size, self.cluster_size_sum, topk_idx[:, :10])



        # group_emb, topk_idx, topk_dis, top1_emb = self._get_topK_emb(distances)
        his_group_emb = (his_group_emb - his_embedding).detach() + his_embedding

        return his_group_emb, his_topk_idx, his_distances, his_topK_emb

    def _get_dictionary(self):
        return self.dictionary.data









class InterestDictSoft_cos(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDictSoft_cos, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)
        self.register_buffer('topk_cluster_size_sum', torch.zeros(num_interest))

    def set_decay(self, epoch, curr_step=0, total_step=1000):
        decay = 0.8 + epoch * 0.05
        self.curr_decay = min(decay, self.max_decay)
        print("set decay:{}".format(self.curr_decay))

    def reset_cluster_size(self):
        self.cluster_size_sum.data = torch.zeros_like(self.cluster_size_sum)

    def set_dict_init(self,proto_dict):
        print("set dict init")
        print(proto_dict)
        print(self.dictionary)
        self.dictionary.data = proto_dict
        self.dict_tmp = self.dictionary.clone().detach()
        print("coarse after")
        print(self.dictionary,self.dictionary.shape)

    def _get_topK_emb(self, distances):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=True, dim=-1)
        topk_idx = idx[:, :self.K]
        distance = distence[:, :self.K]
        distance_neg = distence[:, -self.K:]
        print("topk dist")
        print(distance, distance_neg)
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        # topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        # group_emb = topk_emb_sum / self.K

        topk_w = torch.softmax(distance,dim=1)
        group_emb = torch.matmul(topk_w.unsqueeze(1),topk_emb)
        print(group_emb.shape)
        group_emb = group_emb.reshape([distances.shape[0],-1])
        return group_emb, topk_idx, distance

    def forward(self, inputs_flatten):
        # random initial
        if self.num_update == 1:
            # self._init_dict_by_StdMean(inputs_flatten)
            self._init_dict_by_user(inputs_flatten, 0)
            # torch.distributed.barrier()

        print("input,dict")
        print(self.curr_decay)
        print(inputs_flatten)
        print(self.dictionary)

        # euc_distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        #              + torch.sum(self.dictionary.detach() ** 2, dim=1)
        #              - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))
        distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        print("distances")
        print(distances)
        print(torch.max(distances, dim=1))

        group_emb, topk_idx, topk_dis = self._get_topK_emb(distances)
        print("refer3")
        print(self.training)
        if self.training:
            self.set_decay_updates(self.num_update)

            topk_update = 0
            if topk_update:
                tmp_dist = distances.clone()
                tmp_dist[tmp_dist <= 0] = -1e5
                encodings_W = torch.softmax(tmp_dist * 20, dim=1)
                print("encoding_W:{},{}".format(encodings_W, tmp_dist))
                topk_encodings = encodings_W
            else:
                encodings_W = torch.softmax(topk_dis * 20, dim=1)
                encoding_indices = topk_idx
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot

            # encodings = torch.softmax(distances * 20, dim=1)
            encodings = topk_encodings
            print(encodings)
            print(torch.max(encodings, dim=1))

            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            print(torch.max(tmp_sum, dim=1))
            self.cluster_size = torch.sum(tmp_sum, dim=0)  # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            sum_inplace(self.cluster_size_sum, self.cluster_size)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            input_sum = torch.sum(input_sum_tmp.unsqueeze(dim=0), dim=0)  # cluster_num * dim


            input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)
            # input_sum = input_sum / self.cluster_size.unsqueeze(1)
            print(input_sum)
            print("dict")
            print(self.dictionary)
            ema_tensor_inplace(self.dictionary, input_sum, self.max_decay)
            # ema_tensor_mask_group_inplace(self.dictionary, input_sum, cluster_mask, self.max_decay,
            #                               self.cluster_update_num)
            print(self.dictionary)
            print("update dict")
            print(self.cluster_size, self.cluster_size_sum, topk_idx[:, :10])

            # # topk update
            # encodings_weight = torch.softmax(topk_dis * 20, dim=1)
            # encoding_indices = topk_idx
            # topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
            #                              device=inputs_flatten.device)
            # topk_encodings.scatter_(1, encoding_indices, encodings_weight)  # one-hot
            # print("topk encoding")
            # print(torch.max(topk_encodings, dim=1))
            #
            # topk_tmp_sum = torch.sum(topk_encodings, dim=0, keepdim=True)
            # print(torch.max(topk_tmp_sum, dim=1))
            # self.topk_cluster_size = torch.sum(concat_all_gather(topk_tmp_sum), dim=0)  # 1*cluster_num
            #
            # topk_cluster_mask = torch.zeros_like(self.topk_cluster_size)
            # topk_cluster_mask[self.topk_cluster_size != 0] = 1
            #
            # sum_inplace(self.topk_cluster_size_sum, self.topk_cluster_size)
            # # sum_inplace(self.cluster_update_num, cluster_mask)
            #
            # topk_input_sum_tmp = torch.matmul(topk_encodings.t(), inputs_flatten)
            # topk_input_sum = torch.sum(concat_all_gather(topk_input_sum_tmp.unsqueeze(dim=0)),
            #                            dim=0)  # cluster_num * dim
            #
            # topk_input_sum[topk_cluster_mask == 1] = topk_input_sum[topk_cluster_mask == 1] / self.topk_cluster_size[
            #     topk_cluster_mask == 1].unsqueeze(1)
            #
            # ema_tensor_mask_group_inplace(self.dictionary, input_sum, topk_cluster_mask, self.max_decay,
            #                               self.cluster_update_num)
            # print("topk update dict")
            # print(self.topk_cluster_size, self.topk_cluster_size_sum)
        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten


        return group_emb, topk_idx, distances

    def _get_dictionary(self):
        return self.dictionary.data



class InterestDictSoft_euc(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDictSoft_euc, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)
        self.register_buffer('topk_cluster_size_sum', torch.zeros(num_interest))
    def _get_topK_emb(self, distances):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=False, dim=-1)
        topk_idx = idx[:, :self.K]
        distance = distence[:, :self.K]
        distance_neg = distence[:, -self.K:]
        print("topk dist")
        print(distance, distance_neg)

        scale_min = torch.min(distence)
        scale_max = torch.max(distence)
        norm_dist = 1 - (distence - scale_min) / (scale_max - scale_min)

        norm_distance = norm_dist[:, :self.K]
        norm_distance_neg = norm_dist[:, -self.K:]
        print("norm topk dist")
        print(norm_distance, norm_distance_neg)

        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        group_emb = topk_emb_sum / self.K
        return group_emb, topk_idx

    def forward(self, inputs_flatten):
        # random initial
        if self.num_update == 1:
            # self._init_dict_by_StdMean(inputs_flatten)
            self._init_dict_by_user(inputs_flatten, 0)
            torch.distributed.barrier()

        print("input,dict")
        print(inputs_flatten)
        print(self.dictionary)

        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.dictionary.detach() ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))
        # distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        distances = nn.functional.normalize(distances, p=2, dim=1)
        # print("distances")
        # print(distances)
        # print(torch.min(distances, dim=1))

        group_emb, topk_idx = self._get_topK_emb(distances)


        if self.training:
            self.set_decay_updates(self.num_update)

            # distances = nn.functional.normalize(distances, p=2, dim=1)
            scale_min = torch.min(distances)
            scale_max = torch.max(distances)
            distances = 1 - (distances - scale_min) / (scale_max - scale_min)
            # print(distances)

            # softmax 权重
            tmp_dist = distances.clone()
            tmp_dist[tmp_dist <= 0.5] = -1e5

            # 聚类统计更新数量
            tmp_cluster_size = distances.clone()
            tmp_cluster_size[tmp_cluster_size <= 0.5] = 0

            tmp_sum = torch.sum(tmp_cluster_size, dim=0, keepdim=True)
            # print("tmp:{}".format(torch.max(tmp_sum, dim=1)))
            self.cluster_size = torch.sum(concat_all_gather(tmp_sum), dim=0)  # 1*cluster_num
            sum_inplace(self.cluster_size_sum, self.cluster_size)
            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1
            print("length > 0:{}".format(len(cluster_mask[cluster_mask != 0][0])))


            encodings = tmp_dist
            # sum_inplace(self.cluster_update_num, cluster_mask)
            encodings_W = torch.softmax(encodings.t() * 20, dim=1)
            print("encoding_W:{},{}".format(encodings_W, encodings.t()))
            input_sum_tmp = torch.matmul(encodings_W, inputs_flatten)
            print("input_sum_tmp:{}".format(input_sum_tmp))
            input_sum = torch.mean(concat_all_gather(input_sum_tmp.unsqueeze(dim=0)), dim=0)  # cluster_num * dim
            print("input_sum:{}".format(input_sum))
            # input_sum = input_sum / self.cluster_size.unsqueeze(1)
            # input_sum = input_sum / self.cluster_size.unsqueeze(1)

            print("dict")
            print(self.dictionary)
            # ema_tensor_inplace(self.dictionary, input_sum, self.max_decay)

            ema_tensor_mask_group_inplace(self.dictionary, input_sum, cluster_mask, self.max_decay,
                                          self.cluster_update_num)
            print(self.dictionary)
            print("update dict")
            print(self.cluster_size, self.cluster_size_sum, topk_idx[:, :10])
            # print(topk_idx[:,:10])



        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten


        return group_emb, topk_idx, distances

class InterestDictSoft_euc2(InterestDict):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDictSoft_euc2, self).__init__(num_interest, dim_interest, decay, max_decay, eps, topK)
        self.register_buffer('topk_cluster_size_sum', torch.zeros(num_interest))
        self.dict_tmp = self.dictionary.clone().detach()

    def set_decay(self,epoch, curr_step=0, total_step=1000):
        decay = 0.6 + epoch * 0.1 + 0.1/total_step * curr_step
        self.curr_decay = min(decay, self.max_decay)
        print("set decay:{}".format(self.curr_decay))

    def set_dict_init(self,proto_dict):
        print("set dict init")
        print(proto_dict)
        print(self.dictionary)
        self.dictionary.data = proto_dict[0]
        self.dict_tmp = self.dictionary.clone().detach()
        print("after")
        print(self.dictionary)

    def _get_topK_emb(self, distances):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=True, dim=-1)
        topk_idx = idx[:, :self.K]
        distance = distence[:, :self.K]
        distance_neg = distence[:, -self.K:]
        print("topk dist")
        print(distance, distance_neg)
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        # topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        # group_emb = topk_emb_sum / self.K

        topk_w = torch.softmax(distance,dim=1)
        group_emb = torch.matmul(topk_w.unsqueeze(1),topk_emb)
        print(group_emb.shape)
        group_emb = group_emb.reshape([distances.shape[0],-1])
        return group_emb, topk_idx, distance

    def forward(self, inputs_flatten):
        # random initial
        # if self.num_update == 1:
        #     # self._init_dict_by_StdMean(inputs_flatten)
        #     self._init_dict_by_user(inputs_flatten, 0)
        #     torch.distributed.barrier()

        print("input,dict")
        print(self.curr_decay)
        print(inputs_flatten)
        print(self.dictionary)

        euc_distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.dictionary.detach() ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))
        distances = nn.CosineSimilarity(dim=2)(inputs_flatten.unsqueeze(1), self.dictionary.unsqueeze(0))
        # distances = nn.functional.normalize(euc_distances, p=2, dim=1)
        # distances = 1-distances
        print("distances")
        print(distances,euc_distances)
        print(torch.max(distances, dim=1))

        group_emb, topk_idx, topk_dis = self._get_topK_emb(distances)
        if self.training:
            # self.set_decay_updates(self.num_update)

            topk_update = 0
            if topk_update:
                tmp_dist = distances.clone()
                tmp_dist[tmp_dist <= 0.5] = -1e5
                encodings_W = torch.softmax(tmp_dist * 20, dim=1)
                print("encoding_W:{},{}".format(encodings_W, tmp_dist))
                topk_encodings = encodings_W
            elif True:
                encodings_W = torch.softmax(topk_dis * 20, dim=1)
                encoding_indices = topk_idx
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot
            else:
                encodings_W = torch.softmax(topk_dis * 20, dim=1)
                encodings_W = encodings_W[:,:1]
                encoding_indices = topk_idx[:, :1]
                print("encoding_W:{},{}".format(encodings_W, topk_dis))
                topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                             device=inputs_flatten.device)
                topk_encodings.scatter_(1, encoding_indices, encodings_W)  # one-hot

            # encodings = torch.softmax(distances * 20, dim=1)
            encodings = topk_encodings
            print(encodings)
            print(torch.max(encodings, dim=1))

            self.cluster_size = torch.sum(encodings, dim=0, keepdim=True)
            print(torch.max(self.cluster_size, dim=1))
            # self.cluster_size = concat_all_gather(tmp_sum)  # 1*cluster_num

            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            cluster_size_all = torch.sum(concat_all_gather(self.cluster_size),dim=0)
            sum_inplace(self.cluster_size_sum, cluster_size_all)
            # sum_inplace(self.cluster_update_num, cluster_mask)

            input_sum = torch.matmul(encodings.t(), inputs_flatten).unsqueeze(0)


            input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)

            # input_sum = input_sum / self.cluster_size.unsqueeze(1)
            print(input_sum,input_sum.shape)
            print("dict")
            print(self.dictionary)
            input_sum_all = concat_all_gather(input_sum)  # w * cluster_num * dim
            cluster_mask_all = concat_all_gather(cluster_mask)
            for input_sum_one,cluster_mask_one in zip(input_sum_all,cluster_mask_all):
                # ema_tensor_inplace(self.dictionary, input_sum_one, self.curr_decay)
                ema_tensor_mask_group_inplace(self.dictionary, input_sum_one, cluster_mask_one, self.curr_decay,self.cluster_update_num)
                print("dist dict:{},{}".format(input_sum_one,cluster_mask_one))
            print("dict updated:{}".format(self.dictionary))
            print("update dict")
            print(self.cluster_size, self.cluster_size_sum, topk_idx[:, :10])

            # # topk update
            # encodings_weight = torch.softmax(topk_dis * 20, dim=1)
            # encoding_indices = topk_idx
            # topk_encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
            #                              device=inputs_flatten.device)
            # topk_encodings.scatter_(1, encoding_indices, encodings_weight)  # one-hot
            # print("topk encoding")
            # print(torch.max(topk_encodings, dim=1))
            #
            # topk_tmp_sum = torch.sum(topk_encodings, dim=0, keepdim=True)
            # print(torch.max(topk_tmp_sum, dim=1))
            # self.topk_cluster_size = torch.sum(concat_all_gather(topk_tmp_sum), dim=0)  # 1*cluster_num
            #
            # topk_cluster_mask = torch.zeros_like(self.topk_cluster_size)
            # topk_cluster_mask[self.topk_cluster_size != 0] = 1
            #
            # sum_inplace(self.topk_cluster_size_sum, self.topk_cluster_size)
            # # sum_inplace(self.cluster_update_num, cluster_mask)
            #
            # topk_input_sum_tmp = torch.matmul(topk_encodings.t(), inputs_flatten)
            # topk_input_sum = torch.sum(concat_all_gather(topk_input_sum_tmp.unsqueeze(dim=0)),
            #                            dim=0)  # cluster_num * dim
            #
            # topk_input_sum[topk_cluster_mask == 1] = topk_input_sum[topk_cluster_mask == 1] / self.topk_cluster_size[
            #     topk_cluster_mask == 1].unsqueeze(1)
            #
            # ema_tensor_mask_group_inplace(self.dictionary, input_sum, topk_cluster_mask, self.max_decay,
            #                               self.cluster_update_num)
            # print("topk update dict")
            # print(self.topk_cluster_size, self.topk_cluster_size_sum)
        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten


        return group_emb, topk_idx, distances

    def _get_dictionary(self):
        return self.dictionary.data
