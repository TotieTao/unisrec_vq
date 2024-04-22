import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm

from utils import check_path, set_device, load_plm, amazon_dataset2fullname

# def load_ratings(file):
#     users, items, inters = set(), set(), set()
#     with open(file, 'r') as fp:
#         for line in tqdm(fp, desc='Load ratings'):
#             try:
#                 item, user, rating, time = line.strip().split(',')
#                 users.add(user)
#                 items.add(item)
#                 inters.add((user, item, float(rating), int(time)))
#             except ValueError:
#                 print(line)
#     return users, items, inters

"""
显示当日08:00:00对应的时间戳
 timestamp  :  time
201407-201412 : 1404172800-1419984000
201501-201503 : 1420070400-1427760000
201501-201506 : 1420070400-1435622400

201607-201701 : 1467302400-1483200000
201701-201704 : 1483200000-1490976000
201701-201707 : 1483200000-1498838400

201601-201701 : 1451577600-1483200000
201701-201707 : 1483200000-1498838400
201701-201801 : 1483200000-1514736000
"""


def n_load_ratings(file):
    users, items, inters_h, inters_l, inters_s, users_h, users_s, users_l = set(), set(), set(), set(), set(), set(), set(), set()
    with gzip.open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load metas'):
            data = json.loads(line)
            user = data['reviewerID']
            item = data['asin']
            time = data['unixReviewTime']
            if 1451577600 <= time < 1483200000:
                users_h.add(user)
                items.add(item)
                inters_h.add((user, item, float(data['overall']), int(time)))
            elif 1483200000 <= time < 1514736000:
                users_l.add(user)
                items.add(item)
                inters_l.add((user, item, float(data['overall']), int(time)))
                if 1483200000 <= time < 1498838400:
                    users_s.add(user)
                    inters_s.add((user, item, float(data['overall']), int(time)))
    users = users_h & users_l & users_s
    print("user count: {}".format(len(users)))
    return users, items, inters_h, inters_s, inters_l


def load_meta_items(file):
    items = set()
    with gzip.open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load metas'):
            data = json.loads(line)
            items.add(data['asin'])
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, userlist, can_items=None, user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []
    user2count = 0
    item2count = 0

    # filter by meta items ， user 和 item都存在
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items and unit[0] in userlist:
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                  % (idx, len(inters), len(user2count), len(item2count)))

    userlist = set(user2count.keys())
    return inters, userlist


def make_inters_in_order(inters, userlist):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user in userlist:
            user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def merge_inters_in_order(inters_h, inters_s, inters_l, userlist):
    user2inters, new_inters, user2inters_h, new_inters_h, user2inters_s, new_inters_s, user2inters_l, new_inters_l = collections.defaultdict(
        list), list(), collections.defaultdict(list), list(), collections.defaultdict(
        list), list(), collections.defaultdict(list), list()
    user2inters_set = collections.defaultdict()
    for inter in inters_h:
        user, item, rating, timestamp = inter
        if user in userlist:
            user2inters[user].append((user, item, rating, timestamp))
            user2inters_h[user].append((user, item, rating, timestamp))

    for inter in inters_s:
        user, item, rating, timestamp = inter
        if user in userlist:
            user2inters[user].append((user, item, rating, timestamp))
            user2inters_s[user].append((user, item, rating, timestamp))

    for inter in inters_l:
        user, item, rating, timestamp = inter
        if user in userlist:
            user2inters_l[user].append((user, item, rating, timestamp))
            user2inters[user].append((user, item, rating, timestamp))


    # sort
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)

    for user in user2inters_h:
        user_inters = user2inters_h[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters_h.append(inter)

    for user in user2inters_l:
        user_inters = user2inters_l[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters_l.append(inter)

    for user in user2inters_s:
        user_inters = user2inters_s[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters_s.append(inter)

    return new_inters, new_inters_h, new_inters_s, new_inters_l


def preprocess_rating(args):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '_5.json.gz')
    rating_users, rating_items, rating_inters_h, rating_inters_s, rating_inters_l = n_load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters_h: ', len(rating_inters_h))
    rating_inters_h, userlist_h = filter_inters(rating_inters_h, rating_users, can_items=meta_items,
                                                user_k_core_threshold=args.user_k,
                                                item_k_core_threshold=args.item_k)
    rating_users = rating_users & userlist_h
    print('The number of raw inters_s: ', len(rating_inters_s))
    rating_inters_s, userlist_s = filter_inters(rating_inters_s, rating_users, can_items=meta_items,
                                                user_k_core_threshold=args.user_k,
                                                item_k_core_threshold=args.item_k)
    rating_users = rating_users & userlist_s
    print('The number of raw inters_l: ', len(rating_inters_l))
    rating_inters_l, userlist_l = filter_inters(rating_inters_l, rating_users, can_items=meta_items,
                                                user_k_core_threshold=args.user_k,
                                                item_k_core_threshold=args.item_k)
    rating_users = rating_users & userlist_l

    # sort interactions chronologically for each user
    rating_inters, rating_inters_h, rating_inters_s, rating_inters_l = merge_inters_in_order(rating_inters_h, rating_inters_s, rating_inters_l, rating_users)
    # rating_inters_h = make_inters_in_order(rating_inters_h, rating_users)
    # rating_inters_s = make_inters_in_order(rating_inters_s, rating_users)
    # rating_inters_l = make_inters_in_order(rating_inters_l, rating_users)

    print('\n')
    print(len(rating_users), len(rating_inters), len(rating_inters_h), len(rating_inters_s), len(rating_inters_l))

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters, rating_inters_h, rating_inters_s, rating_inters_l


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, rating, time = line
        users.add(user)
        items.add(item)
    return users, items


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def generate_text(args, items, features):
    item_text_list = []
    already_items = set()

    dataset_full_name = amazon_dataset2fullname[args.dataset]
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    with gzip.open(meta_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Generate text'):
            data = json.loads(line)
            item = data['asin']
            if item in items and item not in already_items:
                already_items.add(item)
                text = ''
                for meta_key in features:
                    if meta_key in data:
                        meta_value = clean_text(data[meta_key])
                        text += meta_value + ' '
                item_text_list.append([item, text])
    return item_text_list


def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w', encoding='UTF-8') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(item + '\t' + text + '\n')


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items, ['title', 'category', 'brand'])
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])

    return user2items, user2index, item2index


def v1_convert_inters2dict(inters, user2index, item2index):
    user2items = collections.defaultdict(list)
    # user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])

    return user2items, user2index, item2index


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out ， 这部分需要对比实验，看是否需要增强序列
        # train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        train_inters[u_index] = [str(inters[:-2])]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert 3 == len(train_inters[u_index]) + len(valid_inters[u_index]) + len(test_inters[u_index])
        # assert len(user2items[u_index]) == len(train_inters[u_index]) + \
        #        len(valid_inters[u_index]) + len(test_inters[u_index])
    return train_inters, valid_inters, test_inters, user2index, item2index


def v1_generate_training_data(args, rating_inters_h, rating_inters_s, rating_inters_l):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2index, item2index = dict(), dict()
    user2items_h, user2index, item2index = v1_convert_inters2dict(rating_inters_h, user2index, item2index)
    user2items_s, user2index, item2index = v1_convert_inters2dict(rating_inters_s, user2index, item2index)
    user2items_l, user2index, item2index = v1_convert_inters2dict(rating_inters_l, user2index, item2index)

    train_inters_h, valid_inters_h, test_inters_h = dict(), dict(), dict()
    train_inters_s, valid_inters_s, test_inters_s = dict(), dict(), dict()
    train_inters_l, valid_inters_l, test_inters_l = dict(), dict(), dict()

    for u_id in user2index:
        u_index = user2index[u_id]
        inters = user2items_h[u_index]
        # leave one out ， 这部分需要对比实验，看是否需要增强序列
        train_inters_h[u_index] = [str(i_index) for i_index in inters[:-2]]
        # train_inters_h[u_index] = [str(inters[:-2])]
        valid_inters_h[u_index] = [str(inters[-2])]
        test_inters_h[u_index] = [str(inters[-1])]
        # assert 3 == len(train_inters_h[u_index]) + len(valid_inters_h[u_index]) + len(test_inters_h[u_index])
        assert len(user2items_h[u_index]) == len(train_inters_h[u_index]) + \
               len(valid_inters_h[u_index]) + len(test_inters_h[u_index])

    for u_id in user2index:
        u_index = user2index[u_id]
        inters = user2items_s[u_index]
        # leave one out
        train_inters_s[u_index] = [str(i_index) for i_index in inters[:-2]]
        # train_inters_s[u_index] = [str(inters[:-2])]
        valid_inters_s[u_index] = [str(inters[-2])]
        test_inters_s[u_index] = [str(inters[-1])]
        assert len(user2items_s[u_index]) == len(train_inters_s[u_index]) + len(valid_inters_s[u_index]) + len(test_inters_s[u_index])

    for u_id in user2index:
        u_index = user2index[u_id]
        inters = user2items_l[u_index]
        # leave one out
        train_inters_l[u_index] = [str(i_index) for i_index in inters[:-2]]
        # train_inters_l[u_index] = [str(inters[:-2])]
        valid_inters_l[u_index] = [str(inters[-2])]
        test_inters_l[u_index] = [str(inters[-1])]
        assert len(user2items_l[u_index]) == len(train_inters_l[u_index]) + len(valid_inters_l[u_index]) + len(test_inters_l[u_index])

    return train_inters_h, valid_inters_h, test_inters_h, train_inters_s, valid_inters_s, test_inters_s, \
        train_inters_l, valid_inters_l, test_inters_l, user2index, item2index


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def generate_item_embedding(args, item_text_list, item2index, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item2index[item]] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        if word_drop_ratio > 0:
            # print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:, 1:, :].sum(dim=1) / \
                          encoded_sentences['attention_mask'][:, 1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


def convert_to_atomic_files(args, datatype, train_data, valid_data, test_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}' + datatype + '.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}' + datatype + '.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}' + datatype + '.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


def v1_convert_to_atomic_files(args, train_data_h, valid_data_h, test_data_h, train_data_s, valid_data_s, test_data_s, \
                               train_data_l, valid_data_l, test_data_l):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data_h.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}' + '.train.inter'), 'w') as file:
        file.write(
            'user_id:token\titem_id_list:token_seq\titem_id:token\titem_id_list_short:token_seq\titem_id_short:token\titem_id_list_long:token_seq\titem_id_long:token\n')

        for uid in uid_list:
            item_seq_h = train_data_h[uid][-50:-1]
            target_item_h = train_data_h[uid][-1]
            file.write(f'{uid}\t{" ".join(item_seq_h)}\t{target_item_h}\t')

            item_seq_s = train_data_s[uid][-50:-1]
            target_item_s = train_data_s[uid][-1]
            file.write(f'{" ".join(item_seq_s)}\t{target_item_s}\t')

            item_seq_l = train_data_l[uid][-50:-1]
            target_item_l = train_data_l[uid][-1]
            file.write(f'{" ".join(item_seq_l)}\t{target_item_l}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}' + '.valid.inter'), 'w') as file:
        file.write(
            'user_id:token\titem_id_list:token_seq\titem_id:token\titem_id_list_short:token_seq\titem_id_short:token\titem_id_list_long:token_seq\titem_id_long:token\n')

        for uid in uid_list:
            item_seq_h = train_data_h[uid][-50:]
            target_item_h = valid_data_h[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq_h)}\t{target_item_h}\t')

            item_seq_s = train_data_s[uid][-50:]
            target_item_s = valid_data_s[uid][0]
            file.write(f'{" ".join(item_seq_s)}\t{target_item_s}\t')

            item_seq_l = train_data_l[uid][-50:]
            target_item_l = valid_data_l[uid][0]
            file.write(f'{" ".join(item_seq_l)}\t{target_item_l}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}' + '.test.inter'), 'w') as file:
        file.write(
            'user_id:token\titem_id_list:token_seq\titem_id:token\titem_id_list_short:token_seq\titem_id_short:token\titem_id_list_long:token_seq\titem_id_long:token\n')

        for uid in uid_list:
            item_seq_h = (train_data_h[uid] + valid_data_h[uid])[-50:]
            target_item_h = test_data_h[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq_h)}\t{target_item_h}\t')

            item_seq_s = (train_data_s[uid] + valid_data_s[uid])[-50:]
            target_item_s = test_data_s[uid][0]
            file.write(f'{" ".join(item_seq_s)}\t{target_item_s}\t')

            item_seq_l = (train_data_l[uid] + valid_data_l[uid])[-50:]
            target_item_l = test_data_l[uid][0]
            file.write(f'{" ".join(item_seq_l)}\t{target_item_l}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Games',
                        help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--user_k', type=int, default=4, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=4, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../raw/')
    parser.add_argument('--output_path', type=str, default='../downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='../../rs/bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters, rating_inters_h, rating_inters_s, rating_inters_l = preprocess_rating(args)

    # split train/valid/test
    # 后面挑17年历史数据做推荐任务，1415年数据做偏好预测任务, {userid:[itemids]}
    train_inters_h, valid_inters_h, test_inters_h, train_inters_s, valid_inters_s, test_inters_s, \
        train_inters_l, valid_inters_l, test_inters_l, user2index, item2index = v1_generate_training_data(args, rating_inters_h, rating_inters_s, rating_inters_l)

    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # generate PLM emb and save to file

    generate_item_embedding(args, item_text_list, item2index, plm_tokenizer, plm_model, word_drop_ratio=-1)

    # pre-stored word drop PLM embs
    if args.word_drop_ratio > 0:
        generate_item_embedding(args, item_text_list, item2index, plm_tokenizer, plm_model,
                                word_drop_ratio=args.word_drop_ratio)

    # save interaction sequences into atomic files
    v1_convert_to_atomic_files(args, train_inters_h, valid_inters_h, test_inters_h, train_inters_s, valid_inters_s,
                               test_inters_s, \
                               train_inters_l, valid_inters_l, test_inters_l)

    # save useful data
    # write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))
