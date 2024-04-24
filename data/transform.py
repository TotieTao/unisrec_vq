from logging import getLogger
import random
import torch
from recbole.data.interaction import Interaction


def construct_transform(config):
    if config['unisrec_transform'] is None:
        logger = getLogger()
        logger.warning('Equal transform')
        return Equal(config)
    else:
        str2transform = {
            'plm_emb': PLMEmb
        }
        return str2transform[config['unisrec_transform']](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        return interaction


class PLMEmb:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('PLM Embedding Transform in DataLoader.')
        self.item_drop_ratio = config['item_drop_ratio']
        self.item_drop_coefficient = config['item_drop_coefficient']

    def __call__(self, dataset, interaction):
        '''Sequence augmentation and PLM embedding fetching
        '''
        item_seq_len = interaction['item_length']
        item_seq = interaction['item_id_list']
        item_seq_short = interaction['item_id_list_short']
        item_seq_short_len = torch.count_nonzero(item_seq_short, dim=1)
        item_seq_long = interaction['item_id_list_long']
        item_seq_long_len = torch.count_nonzero(item_seq_long, dim=1)


        plm_embedding = dataset.plm_embedding

        item_emb_seq = plm_embedding(item_seq)
        pos_item_id = interaction['item_id']
        pos_item_emb = plm_embedding(pos_item_id)

        mask_p = torch.full_like(item_seq, 1 - self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)

        # short
        item_emb_seq_short = plm_embedding(item_seq_short)
        pos_item_id_short = interaction['item_id_short']
        pos_item_emb_short = plm_embedding(pos_item_id_short)

        mask_p_s = torch.full_like(item_seq_short, 1 - self.item_drop_ratio, dtype=torch.float)
        mask_s = torch.bernoulli(mask_p_s).to(torch.bool)

        # long
        item_emb_seq_long = plm_embedding(item_seq_long)
        pos_item_id_long = interaction['item_id_long']
        pos_item_emb_long = plm_embedding(pos_item_id_long)

        mask_p_l = torch.full_like(item_seq_long, 1 - self.item_drop_ratio, dtype=torch.float)
        mask_l = torch.bernoulli(mask_p_l).to(torch.bool)

        # Augmentation
        rd = random.random()
        if rd < self.item_drop_coefficient:
            # Item drop
            seq_mask = item_seq.eq(0).to(torch.bool)
            mask = torch.logical_or(mask, seq_mask)
            mask[:,0] = True
            drop_index = torch.cumsum(mask, dim=1) - 1

            item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
            item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
            item_emb_seq_aug = plm_embedding(item_seq_aug)
        else:
            # Word drop
            plm_embedding_aug = dataset.plm_embedding_aug
            full_item_emb_seq_aug = plm_embedding_aug(item_seq)

            item_seq_aug = item_seq
            item_seq_len_aug = item_seq_len
            item_emb_seq_aug = torch.where(mask.unsqueeze(-1), item_emb_seq, full_item_emb_seq_aug)

        # Augmentation short
        rd = random.random()
        if rd < self.item_drop_coefficient:
            # Item drop
            seq_mask_s = item_seq_short.eq(0).to(torch.bool) #
            mask_s = torch.logical_or(mask_s, seq_mask_s)
            mask_s[:, 0] = True
            drop_index = torch.cumsum(mask_s, dim=1) - 1

            item_seq_short_aug = torch.zeros_like(item_seq_short).scatter(dim=-1, index=drop_index, src=item_seq_short)
            item_seq_len_short_aug = torch.gather(drop_index, 1, (item_seq_short_len - 1).unsqueeze(1)).squeeze() + 1
            item_emb_seq_short_aug = plm_embedding(item_seq_short_aug)
        else:
            # Word drop
            plm_embedding_aug = dataset.plm_embedding_aug
            full_item_emb_seq_short_aug = plm_embedding_aug(item_seq_short)

            item_seq_short_aug = item_seq_short
            item_seq_len_short_aug = item_seq_short_len
            item_emb_seq_short_aug = torch.where(mask_s.unsqueeze(-1), item_emb_seq_short, full_item_emb_seq_short_aug)

        # Augmentation long
        rd = random.random()
        if rd < self.item_drop_coefficient:
            # Item drop
            seq_mask_l = item_seq_long.eq(0).to(torch.bool)  #
            mask_l = torch.logical_or(mask_l, seq_mask_l)
            mask_l[:, 0] = True
            drop_index = torch.cumsum(mask_l, dim=1) - 1

            item_seq_long_aug = torch.zeros_like(item_seq_long).scatter(dim=-1, index=drop_index,
                                                                          src=item_seq_long)
            item_seq_len_long_aug = torch.gather(drop_index, 1,
                                                  (item_seq_long_len - 1).unsqueeze(1)).squeeze() + 1
            item_emb_seq_long_aug = plm_embedding(item_seq_long_aug)
        else:
            # Word drop
            plm_embedding_aug = dataset.plm_embedding_aug
            full_item_emb_seq_long_aug = plm_embedding_aug(item_seq_long)

            item_seq_long_aug = item_seq_long
            item_seq_len_long_aug = item_seq_long_len
            item_emb_seq_long_aug = torch.where(mask_l.unsqueeze(-1), item_emb_seq_long, full_item_emb_seq_long_aug)

        interaction.update(Interaction({
            'item_emb_list': item_emb_seq,
            'pos_item_emb': pos_item_emb,
            'item_id_list_aug': item_seq_aug,
            'item_length_aug': item_seq_len_aug,
            'item_emb_list_aug': item_emb_seq_aug,

            'item_emb_list_short': item_emb_seq_short,
            'pos_item_emb_short': pos_item_emb_short,
            'item_length_short': item_seq_short_len,
            'item_id_list_short_aug': item_seq_short_aug,
            'item_length_short_aug': item_seq_len_short_aug,
            'item_emb_list_short_aug': item_emb_seq_short_aug,

            'item_emb_list_long': item_emb_seq_long,
            'pos_item_emb_long': pos_item_emb_long,
            'item_length_long': item_seq_long_len,
            'item_id_list_long_aug': item_seq_long_aug,
            'item_length_long_aug': item_seq_len_long_aug,
            'item_emb_list_long_aug': item_emb_seq_long_aug,
        }))

        return interaction.to(0)
