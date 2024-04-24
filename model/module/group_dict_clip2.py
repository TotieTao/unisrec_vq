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
        if cluster_update_num[idx] <= 10:
            decay = decay * cluster_update_num[idx] / 10
        new_out = new[idx] * (1-decay)
        moving_avg[idx] = moving_avg.data[idx] * decay
        moving_avg[idx] = moving_avg[idx] + new_out.detach()

def sum_inplace(sum_data, new):
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


# def laplace_smoothing_dim(x, n_categories,dim=1, eps=1e-5):
#     return (x + eps) / (x.sum(dim=dim, keepdim=True) + n_categories * eps)


class InterestDict(nn.Module):
    def __init__(self, num_interest, dim_interest, decay=0.05, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDict, self).__init__()
        self._num_interest = num_interest
        self._dim_interest = dim_interest

        dictionary = torch.randn(num_interest, dim_interest, requires_grad=False)
        self.register_buffer('dictionary',dictionary)
        # nn.init.normal_(self.dictionary)
        self.register_buffer('cluster_size', torch.zeros(num_interest))
        self.register_buffer('cluster_size_sum', torch.zeros(num_interest))
        self.register_buffer('cluster_value', torch.zeros(num_interest, dim_interest))
        self.register_buffer('num_update', torch.tensor(1))
        self.register_buffer('cluster_update_num', torch.ones(num_interest)) # group updated counts

        self.decay = decay
        self.eps = eps
        self.curr_decay = self.decay
        self.max_decay = max_decay
        self.K = topK
        self.init_dict_num = 0


    def set_decay_updates(self, num_update):
        self.curr_decay=min(self.decay*num_update, self.max_decay)
        if num_update <1000:
            self.num_update += 1

    def _init_dict_by_StdMean(self,input_flatten):
        # x_std,x_mean = torch.std_mean(torch.mean(input_flatten,dim=0))
        # nn.init.normal_(self.dictionary.detach(),std=float(x_std),mean=float(x_mean))
        # self.cluster_value.data.copy_(self.dictionary.clone())
        nn.init.uniform_(self.dictionary.detach(), a=-1, b=1)
        self.cluster_value.data.copy_(self.dictionary.clone())


    def _init_dict_by_user(self,input_flatten,idx_start):
        input_flatten_all = concat_all_gather(input_flatten.clone())
        input_len = input_flatten_all.shape[0]
        init_embed = [
            torch.mean(
                input_flatten_all[list(np.random.randint(0, input_len, size=2)), :],
                dim=0)
            for i in range(self._num_interest)
        ]
        init_embed = torch.stack(init_embed, dim=0)
        torch.distributed.broadcast(init_embed, src=0)
        self.dictionary.data = init_embed
        self.cluster_value.data.copy_(self.dictionary.clone())

        # input_len = input_flatten.shape[0]
        # self.init_dict_num += input_len
        #
        # if self._num_interest / self.init_dict_num > 1:
        #     self.dictionary.data[idx_start:idx_start + input_len, :] = input_flatten.clone()
        # else:
        #     if idx_start == 0:
        #         self.dictionary.data[idx_start:, :] = input_flatten[:self._num_interest, :].clone()
        #     else:
        #         self.dictionary.data[idx_start:, :] = input_flatten[:self._num_interest % idx_start, :].clone()


    def _get_topK_emb(self,distances):
        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=False, dim=-1)
        topk_idx = idx[:, :self.K]
        topk_emb = self.dictionary.data[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        topk_emb_sum = torch.sum(topk_emb, dim=1)  # b*1*d
        group_emb = topk_emb_sum.clone() / self.K
        return group_emb, topk_idx

    def forward(self, inputs_flatten):
        # random initial
        if self.num_update == 1:
            # self._init_dict_by_StdMean(inputs_flatten)
            self._init_dict_by_user(inputs_flatten, 0)
            torch.distributed.barrier()

        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.dictionary.detach() ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.dictionary.data.t()))

        group_emb, topk_idx = self._get_topK_emb(distances)

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encoding_indices = topk_idx
        encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                device=inputs_flatten.device)
        encodings.scatter_(1, encoding_indices, 1)  # one-hot

        if self.training:
            self.set_decay_updates(self.num_update)

            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            self.cluster_size = torch.sum(concat_all_gather(tmp_sum), dim=0)
            cluster_mask = torch.zeros_like(self.cluster_size)
            cluster_mask[self.cluster_size != 0] = 1

            # sum_inplace(self.cluster_sum, encoding_sum)
            # sum_inplace(self.cluster_size_sum, self.cluster_size)
            sum_inplace(self.cluster_update_num, cluster_mask)

            # ema_tensor_inplace(self.cluster_size_sum, self.cluster_size, self.curr_decay)
            ema_tensor_mask_group_inplace(self.cluster_size_sum, self.cluster_size, cluster_mask, self.max_decay, self.cluster_update_num)
            input_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)

            input_sum = torch.sum(concat_all_gather(input_sum_tmp.unsqueeze(dim=0)), dim=0)
            # ema_tensor_inplace(self.cluster_value, input_sum, self.curr_decay)
            ema_tensor_mask_group_inplace(self.cluster_value, input_sum, cluster_mask, self.max_decay, self.cluster_update_num)

            cluster_size = laplace_smoothing(self.cluster_size_sum, self._num_interest, self.eps) * self.cluster_size_sum.sum()
            # embed_normalized = self.cluster_value / cluster_size.unsqueeze(1)
            embed_normalized = self.cluster_value.clone()
            embed_normalized[cluster_mask == 1] = self.cluster_value[cluster_mask == 1] / cluster_size[cluster_mask == 1].unsqueeze(1)

            world_size = dist.get_world_size()
            dist.all_reduce(self.cluster_value.div_(world_size))
            self.dictionary.data.copy_(embed_normalized)
            # The following code is used for updating 'dictionary' buffer
            # tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            # self.cluster_size = torch.sum(concat_all_gather(tmp_sum), dim=0) # 1*cluster_num
            #
            # cluster_mask = torch.zeros_like(self.cluster_size)
            # cluster_mask[self.cluster_size!=0] = 1
            #
            # sum_inplace(self.cluster_size_sum, self.cluster_size)
            # sum_inplace(self.cluster_update_num, cluster_mask)
            #
            # input_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            # input_sum = torch.sum(concat_all_gather(input_sum_tmp.unsqueeze(dim=0)), dim=0) # cluster_num * dim
            #
            # world_size = dist.get_world_size()
            # input_sum[cluster_mask==1] = input_sum[cluster_mask==1] / self.cluster_size[cluster_mask==1].unsqueeze(1)
            # # ema_tensor_mask_dict_inplace(self.cluster_value, input_sum, cluster_mask, self.curr_decay)
            # ema_tensor_mask_group_inplace(self.cluster_value, input_sum, cluster_mask, self.max_decay, self.cluster_update_num)
            #
            # dist.all_reduce(self.cluster_value.div_(world_size))
            # self.dictionary.data.copy_(self.cluster_value)

        print("update dict")
        print(inputs_flatten, self.dictionary, self.cluster_size, self.cluster_size_sum, self.cluster_update_num)

        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten

        return group_emb, topk_idx

