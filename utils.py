import scipy as sp
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import time
import tensorflow as tf
import re
import collections


def load_data(config):
    user_pos_train = {}
    max_user_num, max_item_num = 0, 0
    with open(config['train_file'])as fin:
        for line in fin:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            max_user_num = max(max_user_num, uid)
            max_item_num = max(max_item_num, iid)
            r = float(line[2])
            if r > 3.99:
                if uid in user_pos_train:
                    user_pos_train[uid].append(iid)
                else:
                    user_pos_train[uid] = [iid]

    user_pos_test = {}
    with open(config['test_file'])as fin:
        for line in fin:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            max_user_num = max(max_user_num, uid)
            max_item_num = max(max_item_num, iid)
            r = float(line[2])
            if r > 3.99:
                if uid in user_pos_test:
                    user_pos_test[uid].append(iid)
                else:
                    user_pos_test[uid] = [iid]
    config['user_num'] = max_user_num + 1
    config['item_num'] = max_item_num + 1
    print("user number: %s" % config['user_num'])
    print("item number: %s" % config['item_num'])
    for uid in range(max_user_num):
        if uid not in user_pos_train:
            user_pos_train[uid] = []
    return user_pos_train, user_pos_test


def load_matrix_from_file(config):
    file_name = config['train_file']
    row_idx = []
    col_idx = []
    data = []
    for line in open(file_name):
        user, item, rating = line.strip().split()
        user, item, rating = int(user), int(item), float(rating)
        if rating < 4.0:
            continue
        row_idx.append(user)
        col_idx.append(item)
        data.append(1)
    return sp.sparse.csc_matrix((data, (row_idx, col_idx)), (config['user_num'], config['item_num']))


def timer(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print('%r %2.2f sec' % (method.__name__, end_time - start_time))
        return result

    return timed


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def output_to_file(buf, fout):
    print(buf)
    fout.write(buf + '\n')
    fout.flush()
