import torch
import torch.nn as nn
from ..module.gather import GatherLayer
from util.register import Register
from util.tensor_op import cos_sim
import math
_loss = Register()


def contrastive_loss(loss_fun, output_embeds, tb_output_embeds, **kwargs):
    return _loss.build(loss_fun, output_embeds, tb_output_embeds, **kwargs)


@_loss("simclr")
def _simclr(output_embeds, tb_output_embeds, temperature=20.0):
    all_embeddings = torch.cat([output_embeds, tb_output_embeds], dim=0)
    similarities = cos_sim(all_embeddings, all_embeddings) * temperature

    batch_size = list(output_embeds.size())[0]
    similarities += torch.diag(torch.ones(batch_size * 2)).to(similarities.device) * -1e30
    labels = torch.cat(
        [
            torch.range(0, batch_size-1, dtype=torch.long) + batch_size,
            torch.range(0, batch_size-1, dtype=torch.long)
        ], dim=0
    )
    return nn.CrossEntropyLoss(ignore_index=-1)(similarities, labels.to(similarities.device))


@_loss("infonce")
def _infonce(output_embeds, tb_output_embeds, temperature=20.0, is_print="None"):
    scores = cos_sim(output_embeds, tb_output_embeds) * temperature
    _labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
    if is_print !="None":
        print("{} infonce loss".format(is_print))
        print(scores)
        print(torch.argmax(scores, dim=1))
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(scores, _labels)
    return loss


# @_loss("clip")
# def _clip(output_embeds, tb_output_embeds, temperature=20.0):
#     scores = cos_sim(output_embeds, tb_output_embeds) * temperature
#     _labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
#     cross_entropy_loss = nn.CrossEntropyLoss()
#     loss = cross_entropy_loss(scores, _labels)
#     loss_ = cross_entropy_loss(scores.transpose(0,1), _labels)
#     loss = (loss + loss_) / 2.
#     return loss


# @_loss("dist_infonce")
# def _dist_infonce(output_embeds, tb_output_embeds, temperature=20.0):
#     # whether to use barrier?
#     # torch.distributed.barrier()
#     # tb_output_embeds = concat_all_gather(tb_output_embeds)
#     # print(tb_output_embeds)
#     tb_output_embeds = GatherLayer.apply(tb_output_embeds)
#     # print("ddd")
#     # print(tb_output_embeds)
#     # print(tb_output_embeds[0].requires_grad)
#     # print(tb_output_embeds[0].grad_fn)
#     tb_output_embeds = torch.cat(tb_output_embeds, dim=0)
#     # print()
#     # print(tb_output_embeds.size())
#
#     scores = cos_sim(output_embeds, tb_output_embeds) * temperature
#     # print(scores)
#     batch_size = len(output_embeds)
#     # print(batch_size)
#     # print(torch.distributed.get_rank())
#     _labels = (torch.tensor(range(batch_size), dtype=torch.long) +
#                batch_size * torch.distributed.get_rank()).to(scores.device) # need to be check
#     # print(_labels)
#     # raise ValueError("ddddddddd")
#
#     cross_entropy_loss = nn.CrossEntropyLoss()
#     loss = cross_entropy_loss(scores, _labels)
#     return loss


@_loss("dist_infonce")
def _dist_infonce(output_embeds, tb_output_embeds, temperature=20.0, is_print="None"):
    # whether to use barrier?
    # torch.distributed.barrier()
    # tb_output_embeds = concat_all_gather(tb_output_embeds)
    # print(tb_output_embeds)
    output_embeds = GatherLayer.apply(output_embeds)
    tb_output_embeds = GatherLayer.apply(tb_output_embeds)
    # print("ddd")
    # print(tb_output_embeds)
    # print(tb_output_embeds[0].requires_grad)
    # print(tb_output_embeds[0].grad_fn)
    output_embeds = torch.cat(output_embeds, dim=0)
    tb_output_embeds = torch.cat(tb_output_embeds, dim=0)
    # print()
    # print(tb_output_embeds.size())

    scores = cos_sim(output_embeds, tb_output_embeds) / temperature
    # print(scores)
    batch_size = len(output_embeds)
    # print(batch_size)
    # print(torch.distributed.get_rank())
    _labels = torch.tensor(range(batch_size), dtype=torch.long, device=scores.device)
    # print(_labels)
    # raise ValueError("ddddddddd")

    if is_print !="None":
        print("{} infonce loss".format(is_print))
        print(scores)
        print(torch.argmax(scores, dim=1))
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(scores, _labels)
    return loss

@_loss("dist_simclr")
def _dist_simclr(output_embeds, tb_output_embeds, temperature=20.0, is_print="None"):
    # whether to use barrier?
    # torch.distributed.barrier()
    # tb_output_embeds = concat_all_gather(tb_output_embeds)
    # print(tb_output_embeds)
    output_embeds = GatherLayer.apply(output_embeds)
    tb_output_embeds = GatherLayer.apply(tb_output_embeds)
    # print("ddd")
    # print(tb_output_embeds)
    # print(tb_output_embeds[0].requires_grad)
    # print(tb_output_embeds[0].grad_fn)
    output_embeds = torch.cat(output_embeds, dim=0)
    tb_output_embeds = torch.cat(tb_output_embeds, dim=0)
    # print()
    # print(tb_output_embeds.size())

    scores = cos_sim(output_embeds, tb_output_embeds) / temperature
    # print(scores)
    batch_size = len(output_embeds)
    # print(batch_size)
    # print(torch.distributed.get_rank())
    _labels = torch.tensor(range(batch_size), dtype=torch.long, device=scores.device)
    # print(_labels)
    # raise ValueError("ddddddddd")

    if is_print !="None":
        print("{} simclr loss".format(is_print))
        print(scores)
        print(torch.argmax(scores, dim=1))
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(scores, _labels)
    return loss

@_loss("dist_hinge")
def _dist_infonce(output_embeds, tb_output_embeds, temperature=20.0, is_print="None"):
    # whether to use barrier?
    # torch.distributed.barrier()
    # tb_output_embeds = concat_all_gather(tb_output_embeds)
    # print(tb_output_embeds)
    output_embeds = GatherLayer.apply(output_embeds)
    tb_output_embeds = GatherLayer.apply(tb_output_embeds)
    # print("ddd")
    # print(tb_output_embeds)
    # print(tb_output_embeds[0].requires_grad)
    # print(tb_output_embeds[0].grad_fn)
    output_embeds = torch.cat(output_embeds, dim=0)
    tb_output_embeds = torch.cat(tb_output_embeds, dim=0)
    # print()
    # print(tb_output_embeds.size())

    scores = cos_sim(output_embeds, tb_output_embeds) / temperature
    # print(scores)
    batch_size = len(output_embeds)
    # print(batch_size)
    # print(torch.distributed.get_rank())
    _labels = torch.tensor(range(batch_size), dtype=torch.long, device=scores.device)
    # print(_labels)
    # raise ValueError("ddddddddd")

    if is_print !="None":
        print("{} hinge loss".format(is_print))
        print(scores)
        print(torch.argmax(scores, dim=1))
    hinge_loss = nn.HingeEmbeddingLoss(margin=2)
    loss = hinge_loss(scores, _labels)
    return loss


# @_loss("cosine_mse")
# def _cosine_mse(output_embeds, tb_output_embeds, labels, dim=1, threshold=0, **kwargs):
#     # print(output_embeds.size())
#     # print(tb_output_embeds.size())
#     similarity = nn.CosineSimilarity(dim=dim)(output_embeds, tb_output_embeds)
#     # print(similarity.size())
#     labels = labels * 2. - 1.
#     predictions = [[0, 1] if sim > threshold else [1, 0] for sim in similarity.cpu().detach().numpy()]
#     return torch.tensor(predictions), nn.MSELoss()(similarity, labels)


# @_loss("cosine_contrastive")
# def _cosine_contrastive(output_embeds, tb_output_embeds, labels, dim=1, margin=-0.9, **kwargs):
#     # print(output_embeds.size())
#     # print(tb_output_embeds.size())
#     similarity = nn.CosineSimilarity(dim=dim)(output_embeds, tb_output_embeds)
#     # print(similarity.size())
#     similarity = torch.stack([-1*similarity, similarity], dim = -1)
#     labels = labels * 2. - 1.
#     return similarity, nn.CosineEmbeddingLoss(margin=margin)(output_embeds, tb_output_embeds, labels)

