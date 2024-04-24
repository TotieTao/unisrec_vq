from torch import nn
import math
from functools import partial
import torch
import os


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
        B, length = key.shape

        x = self.word_embedding(key)  # patch linear embedding
        x = x * value.unsqueeze(-1)
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask = torch.cat([torch.zeros((B, 1), dtype=mask.dtype, device=mask.device), mask], dim=-1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.pos_drop(x), mask

    def forward(self, key, value, mask):
        x, mask = self.prepare_tokens(key, value, mask)
        print("input")
        print(x,mask)
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
        return x[:, 0], torch.cat(xs[-4:], dim=1), atten_map

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
        _out, _outL, att_map = self.backbone(key, value, mask)
        return self.head(_out), _out, _out


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


class DinoModel(nn.Module):
    def __init__(self, flags, run_inference=False, T_t=0.04, S_t=0.1,
                 projection_size=256, norm_last_layer=False, need_grad_proj=False, m=0.999, transformer='small'):
        super(DinoModel, self).__init__()
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
        if transformer == 'small':
            self.target_network = vit_small(vocab_size=flags.vocab_size, drop_path_rate=0.1, need_grad_proj=need_grad_proj)
        elif transformer == 'base':
            self.target_network = vit_base(vocab_size=flags.vocab_size, drop_path_rate=0.1, need_grad_proj=need_grad_proj)
        else:
            self.target_network = vit_tiny(vocab_size=flags.vocab_size, drop_path_rate=0.1, need_grad_proj=need_grad_proj)
        self.target_network = Wrapper(self.target_network, DINOHead(
            embed_dim,
            projection_size,
            use_bn=False,
            norm_last_layer=norm_last_layer,
        ))
        for p in self.target_network.parameters():
            p.requires_grad = False
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_q.data
        self.m = m
        self.T_t = T_t
        self.S_t = S_t
        self.distributed = True
        self.run_inference = run_inference
        self.register_buffer("center", torch.randn(1, projection_size))

    def update_center(self, prob1, prob2):
        if self.distributed:
            with torch.no_grad():
                prob1 = ALL_Gather_Without_GradReduce.apply(prob1)
                prob2 = ALL_Gather_Without_GradReduce.apply(prob2)
                self.center = 0.9 * self.center + 0.1 * torch.cat([prob1, prob2], dim=0).mean(dim=0, keepdim=True)
        else:
            with torch.no_grad():
                self.center = 0.9 * self.center + 0.1 * torch.cat([prob1, prob2], dim=0).mean(dim=0, keepdim=True)

    def forward(self, input, train_step=0, total_step=100000):
        id, key1, value1, mask1, key2, value2, mask2 = input
        if self.run_inference:
            _, feature, _ = self.target_network(key1, value1, mask1)
            emb1 = torch.zeros_like(feature)
            emb2 = torch.zeros_like(feature)
            emb3 = torch.zeros_like(feature)
            emb4 = torch.zeros_like(feature)
            return feature, emb1, emb2, emb3, emb4
        s_output1, out_1, _ = self.online_network(key1, value1, mask1)
        s_output2, out_2, _ = self.online_network(key2, value2, mask2)
        print("s_output")
        print(s_output1,out_1)
        m = self.m + (1 - self.m) * (1 - math.cos(train_step / total_step * math.pi)) * 0.5
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = m * param_k.data + (1-m) * param_q.data

        if train_step < total_step * 0.5:
            self.T_t = 0.04 + (0.07-0.04) * (1 - math.cos(train_step / (total_step*0.5) * math.pi)) * 0.5
        else:
            self.T_t = 0.07

        with torch.no_grad():
            t_output1, cls, _ = self.target_network(key1, value1, mask1)
            t_output2, _, _ = self.target_network(key2, value2, mask2)
            self.update_center(t_output1, t_output2)
            prob1 = t_output1 - self.center
            prob2 = t_output2 - self.center

            t_prob1 = torch.softmax(prob1.div(self.T_t), dim=1)
            t_prob2 = torch.softmax(prob2.div(self.T_t), dim=1)
            print("target net")
            print(t_output1,cls,prob1,t_prob1)
        log_prob1 = torch.log_softmax(s_output1.div_(self.S_t), dim=1)
        log_prob2 = torch.log_softmax(s_output2.div_(self.S_t), dim=1)
        likelihood = - (t_prob1 * log_prob2 + t_prob2 * log_prob1)/2.0
        loss = likelihood.sum(dim=1).mean()
        return _, loss



