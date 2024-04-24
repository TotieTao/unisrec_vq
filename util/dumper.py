# -*- coding: utf-8 -*-#
import torch

import logging
from io import BytesIO


def get_odps_writer(table_name, slice_id):
    import common_io
    return common_io.table.TableWriter(table_name, slice_id=slice_id)


def get_file_writer(file_path):
    return open(file_path, "wt")


def save_checkpoint(bucket, save_path, checkpoint):
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    logging.info('save model to: {}'.format(save_path))
    while True:
        try:
            bucket.put_object(save_path, buffer.getvalue())
        except Exception as inst:
            logging.info("Upload failed, retry.")
            logging.info(inst)
            continue
        break


def load_checkpoint(bucket, model_path, prefix):
    if model_path is not None:
        buffer = BytesIO(bucket.get_object(model_path).read())
        checkpoint = torch.load(buffer, map_location=lambda storage, loc: storage)
        logging.info("{}: Load pre-trained model from {}".format(prefix, model_path))
        return checkpoint
    else:
        logging.info("warning: path for the pre-trained model is not available.")
        return None


def parse_init_trans_checkpoint(checkpoint):
    '''
    transfer names of pre-trained parameters
    :param checkpoint:
    :return:
    '''
    old_keys = []
    new_keys = []
    for key in checkpoint.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        checkpoint[new_key] = checkpoint.pop(old_key)

    # model.load_state_dict(new_checkpoint, strict=False)
    return checkpoint
