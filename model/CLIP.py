from torch import nn
import math
from functools import partial
import torch
import os
from .module.group_dict_ym import InterestDict, InterestDictSoft,InterestDictSoft_cos,InterestDictSoft_euc,InterestDictSoft_euc2,InterestDictSoft_cosv2,InterestDictSoft_coarse
# from .module.group_dict_clip import InterestDict, InterestDictSoft,InterestDictSoft_coarse,InterestDictSoft2
# from .module.group_dict import InterestDict, InterestDictSoft
# from .module.group_dict_clip2 import InterestDict
from util.tensor_op import GeLU
import torch.nn.functional as F
from .loss import contrastive_ccloss
from model.loss import contrastive_loss
from .module.gather import GatherLayer
from util.tensor_op import cos_sim
import torch.distributed as dist
# from .module.data_augmentation import Crop, Mask, Reorder, Random

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        # last_map = x[0]
        # for i in range(1, length):
        #     last_map = torch.matmul(x[i], last_map)
        last_map = x[-1][:,:,0,1:]
        return last_map


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale - mask.unsqueeze(1).unsqueeze(1)*10000.0

        attn = attn.softmax(dim=-1)
        att1 = attn.clone()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, att1


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class Transformer(nn.Module):
    """ Transformer """
    def __init__(self, vocab_size=16384, embedding_size=64, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,  **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.word_embedding = torch.nn.Embedding(vocab_size+1, embedding_size, padding_idx=vocab_size, sparse=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.part_select = Part_Attention()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, key, value, mask):
        B, length, w = key.shape
        # value_m = torch.div(value,torch.count_nonzero(value, dim=2).unsqueeze(-1) + 1e-5)
        # value = value_m * value
        x = self.word_embedding(key)  # patch linear embedding [b, i, w, d]
        x = x * value.unsqueeze(-1) # [b, i, w, 1]
        x = torch.sum(x,dim=2) # [b,i,d]
        x = x / (torch.count_nonzero(value, dim=2).unsqueeze(-1) + 1e-5)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask = torch.cat([torch.zeros((B, 1), dtype=mask.dtype, device=mask.device), mask], dim=-1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.pos_drop(x), mask

    def forward(self, key, value, mask):
        x, mask = self.prepare_tokens(key, value, mask)
        torch.cuda.synchronize()
        xs = []
        atts = []
        for blk in self.blocks:
            xs.append(x[:, 0])
            x, att = blk(x, mask)
            atts.append(att)
        x = self.norm(x)
        xs.append(x[:, 0])
        atten_map = self.part_select(atts)
        return x[:, 0], x, torch.cat(xs[-4:], dim=1), atten_map

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(vocab_size=16384, **kwargs):
    model = Transformer(
        vocab_size=vocab_size, embedding_size=192, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_single(vocab_size=16384, **kwargs):
    model = Transformer(
        vocab_size=vocab_size, embedding_size=768, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(vocab_size=16384, **kwargs):
    model = Transformer(
        vocab_size=vocab_size, embedding_size=384, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(vocab_size=16384, **kwargs):
    model = Transformer(
        vocab_size=vocab_size, embedding_size=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class Wrapper(nn.Module):
    def __init__(self, backbone, head):
        super(Wrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
    def forward(self, key, value, mask):
        cls, x, _outL, att_map = self.backbone(key, value, mask)
        out = torch.mean(x[:,1:],dim=1)
        # cls = nn.functional.normalize(cls, p=2, dim=1)
        # out = nn.functional.normalize(out, p=2, dim=1)
        return self.head(out), cls, out


def all_gather(input_):
    """Gather tensors and concatinate along the first dimension."""
    # Size and dimension.
    # Bypass the function if we are using only 1 GPU
    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world_size == 1:
        return input_

    # Gather tensor
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_.clone()
    torch.distributed.all_gather(tensor_list, input_)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=0).contiguous()

    return output


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def ensure_rank(rank, all_dim):
    assert rank <= all_dim, '{} should smaller or equal to {}'.format(rank, all_dim)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor(tensor, num_partitions, split_rank,
                 contiguous_split_chunks=False):
    """Split a tensor """
    # Get the size and dimension.
    all_dim = tensor.dim() - 1
    ensure_rank(split_rank, all_dim)
    dim_size = divide(tensor.size()[split_rank], num_partitions)

    # Split
    tensor_list = torch.split(tensor, dim_size, dim=split_rank)

    # Note: torch.split does not create contiguous tensors by default
    if contiguous_split_chunks:
        return list(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _split(input_):
    """Split the tensor along its first dimension."""
    # Size and dimension.
    # Bypass the function if we are using only 1 GPU

    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world_size == 1:
        return input_
    input_list = split_tensor(input_, world_size, 0)
    output = input_list[rank].contiguous()
    return output


class ALL_Gather_Without_GradReduce(torch.autograd.Function):
    """ Gather the input from model parallel region and concatinate. """

    @staticmethod
    def forward(ctx, input_):
        return all_gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = True, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class CLIPModel(nn.Module):
    def __init__(self, flags, run_inference=False, T_t=0.04, S_t=0.1,
                 projection_size=128, norm_last_layer=False, need_grad_proj=False, m=0.999, transformer='tiny'):
        super(CLIPModel, self).__init__()
        if transformer == 'small':
            self.online_network = vit_small(vocab_size=flags.vocab_size, drop_path_rate=0.1, need_grad_proj=need_grad_proj)
        elif transformer == 'base':
            self.online_network = vit_base(vocab_size=flags.vocab_size, drop_path_rate=0.1, need_grad_proj=need_grad_proj)
        else:
            self.online_network = vit_tiny(vocab_size=flags.vocab_size, drop_path_rate=0.1, need_grad_proj=need_grad_proj)
        embed_dim = self.online_network.embed_dim
        self.online_network = Wrapper(self.online_network, DINOHead(
            embed_dim,
            projection_size,
            use_bn=False,
            norm_last_layer=norm_last_layer,
        ))
        self.distributed = True
        self.run_inference = run_inference
        self.num_group = flags.num_group
        self.K = flags.topK
        self.temperature = flags.temperature
        # self.contrastive_loss = flags.contrastive_loss

        self.group_dict = InterestDictSoft_cosv2(int(flags.short_num_cluster), embed_dim, topK=self.K)
        self.group_dict_coarse = InterestDictSoft_coarse(int(flags.long_num_cluster), embed_dim, topK=self.K)
        self.mlp = Mlp(embed_dim*2,embed_dim,projection_size)
        self.proj_head1 = nn.Sequential(
            nn.Linear(embed_dim, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12),
            # nn.ReLU(inplace=True),
            GeLU(),
            nn.Linear(projection_size, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12)
        )
        self.proj_head1_1 = nn.Sequential(
            nn.Linear(embed_dim, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12),
            # nn.ReLU(inplace=True),
            GeLU(),
            nn.Linear(projection_size, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12)
        )
        self.proj_head1_1.apply(self._init_weights)


        self.proj_head3 = nn.Sequential(
            nn.Linear(embed_dim, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12),
            # nn.ReLU(inplace=True),
            GeLU(),
            nn.Linear(projection_size, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12)
        )
        self.proto_head = nn.Sequential(
            nn.Linear(embed_dim, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12),
            # nn.ReLU(inplace=True),
            GeLU(),
            nn.Linear(projection_size, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12)
        )
        self.proto_head.apply(self._init_weights)

        self.proj_head1.apply(self._init_weights)

        self.proj_head3.apply(self._init_weights)

        # self.ln = nn.LayerNorm(embed_dim)

        self.criterion_instance = contrastive_ccloss.InstanceLoss(flags.batch_size, self.temperature)
        self.criterion_head = contrastive_ccloss.InstanceLoss(flags.batch_size, self.temperature)
        # self.criterion_cluster = contrastive_ccloss.ClusterLoss2(self.num_group, 1)
        self.criterion_cluster = contrastive_ccloss.ClusterLoss(self.num_group, self.temperature)

        self.regloss = EmbLoss()

        self.proto_coarse_head = nn.Sequential(
            nn.Linear(embed_dim, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12),
            # nn.ReLU(inplace=True),
            GeLU(),
            nn.Linear(projection_size, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-12)
        )
        self.proto_coarse_head.apply(self._init_weights)

        self.proj_head2 = nn.Sequential(
            nn.Linear(embed_dim, projection_size, bias=True),
            nn.LayerNorm(projection_size, eps=1e-6),
            # nn.ReLU(inplace=True),
            GeLU(),
            # nn.ReLU(),
            nn.Linear(projection_size, self.num_group, bias=True),
            nn.LayerNorm(self.num_group, eps=1e-6),
        )
        self.proj_head2.apply(self._init_weights)

        # self.prototypes = nn.Linear(embed_dim, self.num_group, bias=False)
        # self.prototypes.apply(self._init_weights)
        #
        # self.beta_proj = nn.Linear(embed_dim,self.num_group)
        # self.beta_proj2 = nn.Linear(embed_dim,embed_dim)
        #
        # self.beta_proj_coarse = nn.Linear(embed_dim, int(self.num_group/4))
        # self.beta_proj_coarse.apply(self._init_weights)
        #
        # self.beta_proj.apply(self._init_weights)
        # self.beta_proj2.apply(self._init_weights)
        # self.init_dict = True



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def back_hook(self, grad):
        print("clip hook in:", grad)
    def back_hook1(self, grad):
        print("vq hook in:", grad)

    def neloss(self,c_i,c_j):
        c_i = GatherLayer.apply(c_i)
        c_j = GatherLayer.apply(c_j)
        c_i = torch.cat(c_i, dim=0)
        c_j = torch.cat(c_j, dim=0)
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j
        print("neloss")
        print(ne_i,ne_j,p_i,p_j)
        return ne_loss

    def protoNCE(self,logits,labels):
        _labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits, _labels)
        world_size = dist.get_world_size()
        dist.all_reduce(loss.div_(world_size))
        return loss
    # def _one_pair_data_augmentation(self, keys, values, masks):
    #     """
    #     provides two positive samples given one sequence
    #     """
    #     batch_size,l = keys.shape
    #     augmented_seqs = []
    #     data_tmp = [list(zip(keys[i],values[i])) for i in range(batch_size)]
    #
    #     for i in range(batch_size):
    #         len_key = masks[i]
    #
    #     for i in range(2):
    #         augmented_input_ids = self.base_transform(input_ids)
    #         pad_len = self.max_len - len(augmented_input_ids)
    #         augmented_input_ids = [0] * pad_len + augmented_input_ids
    #
    #         augmented_input_ids = augmented_input_ids[-self.max_len :]
    #
    #         assert len(augmented_input_ids) == self.max_len
    #
    #         cur_tensors = torch.tensor(augmented_input_ids, dtype=torch.long)
    #         augmented_seqs.append(cur_tensors)
    #     return augmented_seqs

    def forward(self, input, proto_dict=[], train_epoch=0, curr_step=0, total_step=100000, update_dict=False,
                run_infer="run_infer1"):
        id, key1, value1, mask1, key2, value2, mask2, key3, value3, mask3 = input
        pretrain_epoch = 0
        # if self.run_inference:
        if run_infer=="run_infer1":
            print("CLIP eval:{}".format(self.training))
            out_head, cls, out_pooling = self.online_network(key1, value1, mask1)
            pooling_proj = self.proj_head3(out_pooling)
            cls_proj = self.proj_head1(cls)
            if train_epoch < pretrain_epoch:
                return cls, out_pooling, cls, cls_proj, pooling_proj

            print("refer1")
            group_emb, topk_idx, sim_mtx, topK_emb = self.group_dict(None,out_pooling)
            print("refer2")
            group_emb_coarse, topk_idx_coarse, sim_mtx_coarse, top1_coarse_emb = self.group_dict_coarse(None, out_pooling)

            # group_emb_mix = topK_emb.reshape([group_emb.shape[0], -1])
            #
            # group_cat_emb_proj = self.mlp(group_emb_mix)

            # group_emb_mix = group_emb + group_emb_coarse
            # group_emb_proj = self.mlp(group_emb_mix)

            # group_emb, topk_idx = self.group_dict(cls)
            # new_grpemb = group_emb.reshape([cls.shape[0],-1])

            # topk3_emb1, topk5_emb1, topk10_emb1, topk_idx1 = self.group_dict(cls)
            # new_grpemb = torch.cat((topk3_emb1, topk5_emb1, topk10_emb1), dim=1)

            group_emb_proj = self.proj_head1_1(group_emb)
            group_emb_proj_coarse = self.proto_coarse_head(group_emb_coarse)
            # group_emb_proj = self.proj_head1_1(group_emb_coarse)

            # cls_proj = self.proj_head1(cls)
            # cls_proj = self.proj_head1(cls)

            # new_grpemb = torch.cat((group_emb, out_pooling), dim=1)
            # user_proj = self.proj_head1(group_emb)

            return cls, out_pooling, group_emb, group_emb_proj, pooling_proj
            # return cls, out_pooling, group_emb, group_emb_proj, group_emb_coarse, group_emb_proj_coarse, pooling_proj
            # return cls, out_pooling, cls_proj,pooling_proj,group_emb
        elif run_infer=="run_infer2":
            out_head, cls, out_pooling = self.online_network(key2, value2, mask2)
            pooling_proj = self.proj_head3(out_pooling)
            cls_proj = self.proj_head1(cls)

            return cls, out_pooling, cls, cls_proj, pooling_proj
        elif run_infer=="run_infer3":
            out_head, cls, out_pooling = self.online_network(key3, value3, mask3)
            pooling_proj = self.proj_head3(out_pooling)
            cls_proj = self.proj_head1(cls)

            return cls, out_pooling, cls, cls_proj, pooling_proj
        elif run_infer=="eval":
            out_head, cls, out_pooling = self.online_network(key1, value1, mask1)
            pooling_proj = self.proj_head3(out_pooling)
            # cls_proj = self.proj_head1(cls)
            group_emb, topk_idx, sim_mtx, topK_emb = self.group_dict(None, out_pooling)
            group_emb_coarse, topk_idx_coarse, sim_mtx_coarse, top1_coarse_emb = self.group_dict_coarse(None,out_pooling)

            group_emb_mix1 = torch.cat((group_emb,group_emb_coarse),dim=-1)
            group_cat_emb_proj1 = self.mlp(group_emb_mix1)

            group_emb_proj = self.proj_head1_1(group_emb)
            group_emb_proj_coarse = self.proto_coarse_head(group_emb_coarse)
            return group_cat_emb_proj1, out_pooling, group_emb, group_emb_proj, group_emb_coarse, group_emb_proj_coarse, topk_idx[:,0]



        _, cls1, out_pooling1 = self.online_network(key1, value1, mask1)
        _, cls2, out_pooling2 = self.online_network(key2, value2, mask2)
        _, cls3, out_pooling3 = self.online_network(key3, value3, mask3)
        # print(out_pooling1)
        # group_proj1 = self.proj_head2(out_pooling1)
        # group_proj2 = self.proj_head2(out_pooling2)
        # group_prob1 = torch.softmax(group_proj1, dim=1)
        # group_prob2 = torch.softmax(group_proj2, dim=1)
        # group_loss = self.criterion_cluster(group_prob1, group_prob2) * 0.1

        user_proj_h = self.proj_head3(out_pooling1)
        user_proj_s = self.proj_head3(out_pooling2)
        user_proj_l = self.proj_head3(out_pooling3)

        # user_proj_h = self.proj_head3(cls1)
        # user_proj_s = self.proj_head3(cls2)
        # user_proj_l = self.proj_head3(cls3)

        short_poolingloss1 = contrastive_loss('dist_infonce', user_proj_h, user_proj_s,
                                              temperature=self.temperature,
                                              is_print="poolingCL")
        short_poolingloss2 = contrastive_loss('dist_infonce', user_proj_s, user_proj_h,
                                              temperature=self.temperature)
        short_poolingloss = (short_poolingloss1 + short_poolingloss2) / 2

        long_poolingloss1 = contrastive_loss('dist_infonce', user_proj_h, user_proj_l, temperature=self.temperature,
                                             is_print="poolingCL")
        long_poolingloss2 = contrastive_loss('dist_infonce', user_proj_l, user_proj_h, temperature=self.temperature)
        long_poolingloss = (long_poolingloss1 + long_poolingloss2) / 2

        poolingloss = short_poolingloss + long_poolingloss


        # poolingloss = torch.zeros(1, device=cls1.device)
        #
        # user_proj1 = self.proj_head1(cls1)
        # user_proj2 = self.proj_head1(cls2)
        # clsloss1 = contrastive_loss('dist_infonce', user_proj1, user_proj2, temperature=self.temperature,
        #                             is_print="clsCL")
        # clsloss2 = contrastive_loss('dist_infonce', user_proj2, user_proj1, temperature=self.temperature)
        # clsloss = (clsloss1 + clsloss2) / 2
        clsloss = torch.zeros(1, device=cls1.device)

        # clsloss = self.criterion_instance(user_proj1,user_proj2)
        # poolingloss = self.criterion_instance(user_proj_h,user_proj_s)

        # beta1 = self.beta_proj(out_pooling1)
        # group_prob1 = torch.softmax(beta1, dim=1)
        # beta2 = self.beta_proj(out_pooling2)
        # group_prob2 = torch.softmax(beta2, dim=1)
        # neloss = self.neloss(group_prob1, group_prob2)

        if update_dict:
            # self.group_dict.set_decay(10)
            self.group_dict.set_decay(train_epoch, curr_step, total_step)
            self.group_dict_coarse.set_decay(train_epoch, curr_step, total_step)

            if train_epoch >= pretrain_epoch and train_epoch < 3 and curr_step == 0:
                print("proto len:{},{},{}".format(len(proto_dict), len(proto_dict[0]), len(proto_dict[1])))
                self.group_dict.set_dict_init(proto_dict[1])
                self.group_dict_coarse.set_dict_init(proto_dict[0])

            if curr_step == 0:
                self.group_dict.reset_cluster_size()
                self.group_dict_coarse.reset_cluster_size()

            # group_dict = self.group_dict._get_dictionary()

            # # # #
            # beta1 = self.beta_proj(out_pooling1)
            # group_prob1 = torch.softmax(beta1, dim=1)
            # group_label1 = torch.argmax(group_prob1,dim=-1)
            # #
            # beta2 = self.beta_proj(out_pooling2)
            # group_prob2 = torch.softmax(beta2, dim=1)
            # group_label2 = torch.argmax(group_prob2,dim=-1)

            # # #
            # # # eps1 = torch.randn_like(out_pooling1)
            # # # eps2 = torch.randn_like(out_pooling2)
            # # #
            # # # user_z = self.proj_head2
            # #
            # #
            # # #
            # # neloss = self.neloss(group_prob1,group_prob2)
            # #
            # user_g1 = torch.matmul(user_proj_h,group_dict)
            # user_g2 = torch.matmul(user_proj_s,group_dict)
            # # rec_user1 = self.beta_proj2(user_g1)
            # # rec_user2 = self.beta_proj2(user_g2)
            # # recloss1 = contrastive_loss('dist_hinge', rec_user1, user_proj_h,temperature=self.temperature, is_print="recloss")
            # # recloss2 = contrastive_loss('dist_hinge', rec_user2, user_proj_s,temperature=self.temperature, is_print="recloss")
            # # recloss = recloss1 + recloss2
            # #
            # #
            # #
            # #
            # sim_out1 = nn.CosineSimilarity(dim=2)(out_pooling1.unsqueeze(1), group_dict.unsqueeze(0))
            # sim_out2 = nn.CosineSimilarity(dim=2)(out_pooling2.unsqueeze(1), group_dict.unsqueeze(0))

            # sim_out1 = torch.matmul(out_pooling1, group_dict.T)
            # sim_out2 = torch.matmul(out_pooling2, group_dict.T)

            # # sim_out1 = nn.functional.normalize(sim_out1, p=2, dim=1)
            # # sim_out2 = nn.functional.normalize(sim_out2, p=2, dim=1)
            # # # sim_out2 = sim_out2 / torch.norm(sim_out2, p=2, dim=1).unsqueeze(1)
            # # # sim_out1 = self.prototypes(out_pooling1)
            # # # sim_out2 = self.prototypes(out_pooling2)
            # # #

            # q_sim_out1 = torch.softmax(sim_out1 * 2, dim=1)
            # q_sim_out2 = torch.softmax(sim_out2 * 2, dim=1)
            #
            # q1 = distributed_sinkhorn(sim_out1)
            # q2 = distributed_sinkhorn(sim_out2)
            # print("sim_out")
            # print(sim_out1, q1,q1 * torch.log(q_sim_out2))
            # subloss1 = -torch.mean(torch.sum(q1 * torch.log(q_sim_out2), dim=1))
            # subloss2 = -torch.mean(torch.sum(q2 * torch.log(q_sim_out1), dim=1))
            # group_sub_loss = subloss1 + subloss2
            # world_size = dist.get_world_size()
            # dist.all_reduce(group_sub_loss.div_(world_size))

            # group_sub_loss = torch.zeros(1, device=cls1.device)
            #
            # beta1 = self.beta_proj(out_pooling1)
            # group_prob1 = torch.softmax(beta1, dim=1)
            # beta2 = self.beta_proj(out_pooling2)
            # group_prob2 = torch.softmax(beta2, dim=1)
            # neloss = self.neloss(group_prob1, group_prob2)
            #
            # beta1_coarse = self.beta_proj_coarse(out_pooling1)
            # group_prob1_coarse = torch.softmax(beta1_coarse, dim=1)
            # beta2_coarse = self.beta_proj_coarse(out_pooling2)
            # group_prob2_coarse = torch.softmax(beta2_coarse, dim=1)
            # neloss_coarse = self.neloss(group_prob1_coarse, group_prob2_coarse)

            # neloss_all = neloss + neloss_coarse
            neloss_all = torch.zeros(1, device=cls1.device)

            # mse = nn.MSELoss()

            short_group_emb, short_topk_idx, short_sim_mtx, short_top1_emb = self.group_dict(out_pooling2, out_pooling1)
            # group_emb2, topk_idx2, sim_mtx2, top1_emb2 = self.group_dict(out_pooling3)

            long_group_emb, long_topk_idx, long_sim_mtx, long_top1_emb = self.group_dict_coarse(out_pooling3,out_pooling1)
            # group_emb2_coarse, topk_idx2_coarse, sim_mtx2_coarse, top1_emb2_coarse = self.group_dict_coarse(
            #     out_pooling2)

            #### protonceloss
            # protonce_loss1 = self.protoNCE(F.log_softmax(sim_mtx1,dim=1),group_label1)
            # protonce_loss2 = self.protoNCE(F.log_softmax(sim_mtx2,dim=1),group_label2)
            # protonce_loss = protonce_loss1 + protonce_loss2

            # group_emb_mix1 = torch.cat((short_group_emb,long_group_emb),dim=-1)
            # group_cat_emb_proj1 = self.mlp(group_emb_mix1)


            short_group_emb_proj = self.proj_head1_1(short_group_emb)
            long_group_emb_proj = self.proto_coarse_head(long_group_emb)

            # group_emb_score1 = F.cosine_similarity(group_emb_proj1, group_emb_proj2, dim=1)
            # group_emb_score2 = F.cosine_similarity(group_emb_proj2, group_emb_proj1, dim=1)
            # gt = torch.ones(cls1.shape[0], device=cls1.device)
            # group_emb_loss1 = mse(gt, group_emb_score1)
            # group_emb_loss2 = mse(gt, group_emb_score2)


            group_emb_loss1 = contrastive_loss('dist_infonce', user_proj_h, short_group_emb_proj,
                                               temperature=self.temperature, is_print="groupembCL")
            group_emb_loss2 = contrastive_loss('dist_infonce', short_group_emb_proj, user_proj_h,
                                               temperature=self.temperature)
            short_group_emb_loss = (group_emb_loss1 + group_emb_loss2) / 2

            long_group_emb_loss1 = contrastive_loss('dist_infonce', user_proj_h, long_group_emb_proj,
                                               temperature=self.temperature, is_print="groupembCL")
            long_group_emb_loss2 = contrastive_loss('dist_infonce', long_group_emb_proj, user_proj_h,
                                               temperature=self.temperature)
            long_group_emb_loss = (long_group_emb_loss1 + long_group_emb_loss2) / 2

            group_emb_loss = long_group_emb_loss*0.25 + short_group_emb_loss
############
            # group_emb_loss1 = contrastive_loss('dist_infonce', user_proj_h, group_cat_emb_proj1,
            #                                    temperature=self.temperature, is_print="groupembCL")
            # group_emb_loss2 = contrastive_loss('dist_infonce', group_cat_emb_proj1, user_proj_h,
            #                                    temperature=self.temperature)
            # group_emb_loss = (group_emb_loss1 + group_emb_loss2) / 2

            # group_emb_loss = torch.zeros(1,device=cls1.device)

            short_top1_emb_proj1 = self.proto_head(short_top1_emb)
            # top1_emb_proj2 = self.proto_head(top1_emb2)
            long_top1_emb_proj1 = self.proto_coarse_head(long_top1_emb)
            # topk_coarse_emb_proj2 = self.proto_coarse_head(long_top1_emb)

            # user_loss1 = contrastive_loss('dist_infonce', user_proj_h, group_emb_proj1, temperature=self.temperature,
            #                               is_print="grp_poolingCL")
            # user_loss2 = contrastive_loss('dist_infonce', user_proj_s, group_emb_proj2, temperature=self.temperature)
            # user_loss = user_loss1 + user_loss2
            user_loss = torch.zeros(1, device=cls1.device)

            # proto_score1 = F.cosine_similarity(user_proj_h, group_emb_proj1, dim=1)
            # proto_score2 = F.cosine_similarity(user_proj_s, group_emb_proj2, dim=1)
            # gt = torch.ones(proto_score1.shape[0],device=proto_score1.device)
            # proto_loss1 = mse(gt, proto_score1)
            # proto_loss2 = mse(gt, proto_score2)
            # proto_coarse_score1 = F.cosine_similarity(user_proj_h, topk_coarse_emb_proj1, dim=1)
            # proto_coarse_score2 = F.cosine_similarity(user_proj_s, topk_coarse_emb_proj2, dim=1)
            # gt = torch.ones(cls1.shape[0], device=cls1.device)
            # proto_coarse_loss1 = mse(gt, proto_coarse_score1)
            # proto_coarse_loss2 = mse(gt, proto_coarse_score2)

            # proto_loss1 = F.mse_loss(user_proj_h, short_top1_emb_proj1)
            # proto_loss2 = F.mse_loss(user_proj_h, long_top1_emb_proj1)
            # proto_coarse_loss1 = F.mse_loss(user_proj_h,topk_coarse_emb_proj1)
            # proto_coarse_loss2 = F.mse_loss(user_proj_s,topk_coarse_emb_proj2)
            # protoloss =  (proto_loss1 + proto_loss2) + (proto_coarse_loss1 + proto_coarse_loss2)
            # protoloss = (proto_loss2)
            protoloss = torch.zeros(1,device=cls1.device)

            # protof2c_loss1 = F.mse_loss(group_emb_proj1,topk_coarse_emb_proj1)
            # protof2c_loss2 = F.mse_loss(group_emb_proj2,topk_coarse_emb_proj2)
            # protof2c_loss = protof2c_loss1 + protof2c_loss2

            # protof2c_socre1 = F.cosine_similarity(group_emb_proj1, topk_coarse_emb_proj1, dim=1)
            # protof2c_score2 = F.cosine_similarity(group_emb_proj2, topk_coarse_emb_proj2, dim=1)
            # gt = torch.ones(cls1.shape[0], device=cls1.device)
            # protof2c_loss1 = mse(gt, protof2c_socre1)
            # protof2c_loss2 = mse(gt, protof2c_score2)
            # protof2c_loss = protof2c_loss1 + protof2c_loss2
            protof2c_loss = torch.zeros(1, device=cls1.device)

            # regloss = self.regloss(out_pooling2, out_pooling1, cls1, cls2) * 1e-5
            ccloss =  group_emb_loss  + protoloss * 0.5
            # ccloss = subloss * 0.1 + regloss
            print("loss")
            print(user_loss, group_emb_loss, protoloss)
            # print(subloss1,subloss2, regloss)
            # head_contrastive_loss = head_contrastive_loss * 0
            # ccloss = user_loss
            # ccloss = torch.zeros(1, device=cls1.device)
        elif train_epoch >= 0 and train_epoch < 3:
            print("pretrain")
            print(curr_step)
            if curr_step == 0:
                print("proto len:{},{},{}".format(len(proto_dict), len(proto_dict[0]), len(proto_dict[1])))
                self.group_dict.set_dict_init(proto_dict[1])
                self.group_dict_coarse.set_dict_init(proto_dict[0])

            with torch.no_grad():
                self.group_dict.set_decay(train_epoch, curr_step, total_step)
                self.group_dict_coarse.set_decay(train_epoch, curr_step, total_step)
                _, _, _, _ = self.group_dict(out_pooling2,out_pooling1)
                # _, _, _, _ = self.group_dict(out_pooling2)

                self.group_dict_coarse.set_decay(train_epoch, curr_step, total_step)
                _, _, _, _ = self.group_dict_coarse(out_pooling3,out_pooling1)
                # _, _, _, _ = self.group_dict_coarse(out_pooling2)

            ccloss = torch.zeros(1, device=cls1.device)
        else:
            ccloss = torch.zeros(1, device=cls1.device)

        i2i_loss = poolingloss
        print(clsloss, poolingloss)

        # head_contrastive_loss = torch.zeros(1, device=cls1.device)

        return ccloss, i2i_loss