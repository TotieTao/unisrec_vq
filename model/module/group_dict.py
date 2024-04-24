import torch
import torch.nn as nn
import torch.distributed as dist
from util.tensor_op import concat_all_gather


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))


def ema_tensor_inplace(moving_avg, new, decay):
    new_out = torch.mul(new, 1.0 - decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())


def sum_inplace(sum_data, new):
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


# def laplace_smoothing_dim(x, n_categories,dim=1, eps=1e-5):
#     return (x + eps) / (x.sum(dim=dim, keepdim=True) + n_categories * eps)


class InterestDict(nn.Module):
    def __init__(self, num_interest, dim_interest, decay=0.1, max_decay=0.99, eps=1e-5, topK=1):
        super(InterestDict, self).__init__()
        self._num_interest = num_interest
        self._dim_interest = dim_interest

        embed = torch.randn(num_interest, dim_interest)
        self.register_buffer('embed', embed)
        nn.init.normal_(self.embed)
        self.register_buffer('cluster_size', torch.zeros(num_interest))
        self.register_buffer('cluster_sum', torch.zeros(num_interest))
        self.register_buffer('embed_avg', torch.zeros(num_interest, dim_interest))

        self.decay = decay
        self.eps = eps
        self.curr_decay=self.decay
        self.max_decay=max_decay
        self.K = topK

    def set_decay_updates(self, num_update):
        self.curr_decay=min(self.decay*num_update, self.max_decay)

    def forward(self, inputs_flatten):
        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embed.data ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.embed.data.t()))

        """aggregate topK cluster embdeding to replace original user embedding"""
        distence, idx = torch.sort(distances, descending=False, dim=-1)
        topk_idx = idx[:, :self.K]
        topk_emb = self.embed[topk_idx]  # batchsize, topk_cluster, emb_dim:b*c*d
        topk_emb_sum = torch.sum(topk_emb, dim=1).unsqueeze(1)  # b*1*d
        W_avg = torch.div(topk_emb, topk_emb_sum)  # b*c*d / b*1*d -> b*c*d
        # W_avg = torch.softmax(topk_emb, dim=1)
        emb_avg = torch.mul(W_avg, topk_emb)  # b*c*d (*) b*c*d -> b*c*d
        group_emb = torch.sum(emb_avg, dim=1)  # b*c*d -> b*d

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_interest, dtype=torch.float,
                                device=inputs_flatten.device)
        encodings.scatter_(1, encoding_indices, 1)  # one-hot

        if self.training:
            # The following code is used for updating 'embed' buffer
            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            encoding_sum = torch.sum(concat_all_gather(tmp_sum), dim=0)

            sum_inplace(self.cluster_sum, encoding_sum)
            ema_tensor_inplace(self.cluster_size, encoding_sum, self.curr_decay)

            embed_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            embed_sum = torch.sum(concat_all_gather(embed_sum_tmp.unsqueeze(dim=0)), dim=0)

            ema_tensor_inplace(self.embed_avg, embed_sum, self.curr_decay)

            cluster_size = laplace_smoothing(self.cluster_size, self._num_interest, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

            world_size = dist.get_world_size()
            dist.all_reduce(embed_normalized.div_(world_size))
            self.embed.data.copy_(embed_normalized)

        quantize = torch.matmul(encodings, self.embed)
        #quantize = inputs_flatten
        quantize = (quantize - inputs_flatten).detach() + inputs_flatten
        group_emb = (group_emb - inputs_flatten).detach() + inputs_flatten

        return group_emb, topk_idx