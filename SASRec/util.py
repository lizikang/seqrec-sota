import sys
import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict



def data_partition(fname):
    # read data from file, sort by time for each user
    file_path = 'files/%s/%s.csv' % (fname, fname)
    data = pd.read_csv(file_path, header=0)
    data = data.sort_values(by = ['user', 'time'])

    # map original user id to new user id(1,2,3...)
    user_map, new_id = {}, 1
    user_set = set(data['user'])

    for u in user_set:
        user_map[u] = new_id
        new_id += 1				
	
    data['user'] = data['user'].map(lambda x: user_map[x])
    user_set = set(data['user'])
    usernum = len(user_set)

    # map original item id to new item id(1,2,3...)
    item_map, new_id = {}, 1	
    item_set = set(data['item'])

    for i in item_set:
        item_map[i] = new_id
        new_id += 1				
	
    data['item'] = data['item'].map(lambda x: item_map[x])
    item_set = set(data['item'])
    itemnum = len(item_set)

    # split train/valid/test set
    user_train, user_valid, user_test = {}, {}, {}
    user_list, item_list = list(data['user']), list(data['item'])
    user_ids, indices, counts = np.unique(user_list, return_index = True, return_counts = True)

    for i in range(len(user_ids)):
        uid, index, length = user_ids[i], indices[i], counts[i]
        items = item_list[index: index+length]
        user_train[uid], user_valid[uid], user_test[uid] = items[:-2], [items[-2]], [items[-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]



def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    valid_user = 0.0
    ap = 0.0
    HT_5 = 0.0
    HT_10 = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(500):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1
        ap += 1 / (rank+1)
        if rank < 5:
            HT_5 += 1
            NDCG_5 += 1 / np.log2(rank + 2)
        if rank < 10:
            HT_10 += 1
            NDCG_10 += 1 / np.log2(rank + 2)
        if valid_user % 100 == 0:
            #print('.')
            sys.stdout.flush()
    return ap/valid_user,  HT_5/valid_user, HT_10/valid_user, NDCG_5/valid_user, NDCG_10/valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    valid_user = 0.0
    ap = 0.0
    HT_5 = 0.0
    HT_10 = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(500):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1
        ap += 1 / (rank+1)
        if rank < 5:
            HT_5 += 1
            NDCG_5 += 1 / np.log2(rank + 2)
        if rank < 10:
            HT_10 += 1
            NDCG_10 += 1 / np.log2(rank + 2)
        if valid_user % 100 == 0:
            #print('.')
            sys.stdout.flush()
    return ap/valid_user,  HT_5/valid_user, HT_10/valid_user, NDCG_5/valid_user, NDCG_10/valid_user
