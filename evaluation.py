import numpy as np
import random
import multiprocessing
import time
import tensorflow as tf
from utils import timer, get_current_time


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k, pos_num):
    dcg_max = dcg_at_k([1] * pos_num, k)
    # dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def eval_one_user(inputs, topk=50):
    user, ratings, user_pos_test = inputs
    user_pos_test = set(user_pos_test)

    idx = np.argpartition(ratings, -topk)[-topk:]
    item_sort = list(idx[np.argsort(-ratings[idx])])

    r = []
    mrr = None
    for idx, i in enumerate(item_sort):
        if i in user_pos_test:
            if mrr is None:
                mrr = idx + 1
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    p_50 = np.mean(r[:50])
    pos_num = len(user_pos_test)
    ndcg_3 = ndcg_at_k(r, 3, pos_num)
    ndcg_5 = ndcg_at_k(r, 5, pos_num)
    ndcg_10 = ndcg_at_k(r, 10, pos_num)
    ndcg_50 = ndcg_at_k(r, 50, pos_num)
    mrr = 1. / mrr if mrr is not None else 0
    return np.array([p_3, p_5, p_10, p_50, ndcg_3, ndcg_5, ndcg_10, ndcg_50, mrr])


def eval_one_user_process(inputs):
    user, d_ratings, user_pos_train, user_pos_test = inputs
    for i in user_pos_train:
        d_ratings[i] = -np.inf
    res_dis = eval_one_user([user, d_ratings, user_pos_test])
    return res_dis


def eval(sess, discriminator, user_pos_train, user_pos_test):
    result_dis = np.array([0.] * 9)
    pool = multiprocessing.Pool(5)
    batch_size = 4096
    test_users = list(user_pos_test.keys())
    random.shuffle(test_users)
    test_users = test_users[:50000]
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        batch_user = test_users[index:index + batch_size]
        index += batch_size

        batch_dis_ratings = sess.run(discriminator.get_all_ratings(), feed_dict={discriminator.u: batch_user})
        iter_data = []
        for i in range(len(batch_user)):
            user = batch_user[i]
            iter_data.append((user, batch_dis_ratings[i], user_pos_train[user], user_pos_test[user]))
        all_results = pool.map(eval_one_user_process, iter_data)

        for re in all_results:
            result_dis += re

    pool.close()
    ret_dis = result_dis / test_user_num
    ret_dis = list(ret_dis)
    return ret_dis

