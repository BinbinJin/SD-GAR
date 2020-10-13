import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import random
import math
import time
import sys
import json
import pickle
import multiprocessing

from discriminator import DIS
from generator import GEN
from utils import load_data, get_current_time, output_to_file
from evaluation import eval
from sampling import SamplerD, SamplerG
from alias_method import AliasTable

Gen_data_for_d_time = 0.
Train_d_time = 0.
Train_d_cnt = 0
Gen_data_for_g_time = 0.
Train_g_time = 0.
Train_g_cnt = 0


def get_init_embeddings(config):
    user_num = config['user_num']
    item_num = config['item_num']
    gen_emb_dim = config['gen_emb_dim']
    prob_users = np.random.rand(user_num, gen_emb_dim)
    prob_items = np.random.rand(item_num, gen_emb_dim)
    for u in range(user_num):
        prob_users[u] = prob_users[u] / np.sum(prob_users[u])
    for k in range(gen_emb_dim):
        prob_items[:, k] = prob_items[:, k] / np.sum(prob_items[:, k])
    return prob_users, prob_items


def get_partition_funciton(sess, discriminator, generator, alias_table, config):
    global Train_g_time, Gen_data_for_g_time
    sampler = SamplerG(config, sess, discriminator, alias_table, 'x')

    time_start = time.time()
    random_k, random_j, random_prob = sess.run(generator.get_random_for_item())
    random_dic = {'random_k': random_k, 'random_j': random_j, 'random_prob': random_prob}
    sampler.generate_data(sample_num=config['gen_sample_num_item'], random_dic=random_dic)
    sampler.generate_logits()
    Gen_data_for_g_time += time.time() - time_start

    user_num = config['user_num']
    z_u = np.zeros([user_num], np.float)
    index = 0
    batch_size = config['batch_size']
    time_start = time.time()
    while sampler.index < user_num:
        batch = sampler.get_next_batch(batch_size)
        index_end = min(user_num, index + batch_size)
        batch_users_for_2d = batch['batch_users_for_2d']
        batch_items_for_2d = batch['batch_items_for_2d']
        batch_d_logits = batch['batch_d_logits']
        batch_z_u = sess.run(generator.get_partition(),
                             feed_dict={generator.u_x: batch_users_for_2d,
                                        generator.i_x: batch_items_for_2d,
                                        generator.d_logits_x: batch_d_logits,
                                        generator.user_embeddings: alias_table.prob_users,
                                        generator.item_embeddings: alias_table.prob_items})
        batch_z_u = np.reshape(batch_z_u, [-1])
        z_u[index:index_end] = batch_z_u
        index += batch_size
    Train_g_time += time.time() - time_start
    return z_u, sampler.d_logits


def get_new_user_embeddings(sess, discriminator, generator, z_u, alias_table, config):
    global Train_g_time, Gen_data_for_g_time
    sampler = SamplerG(config, sess, discriminator, alias_table, 'x')

    time_start = time.time()
    random_k, random_j, random_prob = sess.run(generator.get_random_for_item())
    random_dic = {'random_k': random_k, 'random_j': random_j, 'random_prob': random_prob}
    sampler.generate_data(sample_num=config['gen_sample_num_item'], random_dic=random_dic)
    sampler.generate_logits()
    Gen_data_for_g_time += time.time() - time_start

    user_num = config['user_num']
    emb_dim = config['gen_emb_dim']
    user_embeddings = np.zeros([user_num, emb_dim], np.float)
    index = 0
    batch_size = config['batch_size']
    time_start = time.time()
    while sampler.index < user_num:
        batch = sampler.get_next_batch(batch_size)
        index_end = min(user_num, index + batch_size)
        batch_users_for_2d = batch['batch_users_for_2d']
        batch_items_for_2d = batch['batch_items_for_2d']
        batch_d_logits = batch['batch_d_logits']
        batch_z_u = z_u[index:index_end]
        batch_user_embeddings = sess.run(generator.get_new_user_embeddings(),
                                         feed_dict={generator.u_x: batch_users_for_2d,
                                                    generator.i_x: batch_items_for_2d,
                                                    generator.d_logits_x: batch_d_logits,
                                                    generator.z_u_x: batch_z_u,
                                                    generator.user_embeddings: alias_table.prob_users,
                                                    generator.item_embeddings: alias_table.prob_items})
        user_embeddings[index:index_end] = batch_user_embeddings
        index += batch_size
    temp = np.max(user_embeddings, axis=1)
    temp = np.mean(temp) * config['lambda_x']
    user_embeddings = sess.run(generator.user_embeddings_norm,
                               feed_dict={generator.user_embeddings: user_embeddings,
                                          generator.lambda_x: temp})
    Train_g_time += time.time() - time_start
    return user_embeddings


def get_new_item_embeddings(sess, discriminator, generator, z_u, alias_table, config):
    global Train_g_time, Gen_data_for_g_time
    user_num = config['user_num']
    item_num = config['item_num']
    emb_dim = config['gen_emb_dim']
    batch_size = config['batch_size']

    item_embeddings = np.zeros([item_num, emb_dim], np.float)
    for k in range(emb_dim):
        sampler = SamplerG(config, sess, discriminator, alias_table, 'y')

        time_start = time.time()
        # use gpu to sample users
        u_k_prob = alias_table.prob_users[:, k] / np.sum(alias_table.prob_users[:, k])
        at = AliasTable.alias_table_construct(u_k_prob)
        threshold = np.reshape(np.array([threshold for threshold, _ in at]), [1, user_num])
        index_1 = np.reshape(np.arange(user_num), [1, user_num])
        index_2 = np.reshape(np.array([ind if ind is not None else user_num for _, ind in at]), [1, user_num])
        data = sess.run(generator.sample_u, {generator.threshold_user: threshold,
                                             generator.index_1_user: index_1,
                                             generator.index_2_user: index_2})
        sampler.data_user = data
        sampler.data_item = list(range(item_num))
        sampler.generate_logits()
        Gen_data_for_g_time += time.time() - time_start

        index = 0
        time_start = time.time()
        while sampler.index < item_num:
            batch = sampler.get_next_batch(batch_size)
            index_end = min(index + batch_size, item_num)
            batch_users_for_2d = batch['batch_users_for_2d']
            batch_items_for_2d = batch['batch_items_for_2d']
            batch_d_logits = batch['batch_d_logits']
            batch_z_u = z_u[batch_users_for_2d]
            batch_item_embeddings = sess.run(generator.get_new_item_embeddings(),
                                             feed_dict={generator.u_y: batch_users_for_2d,
                                                        generator.i_y: batch_items_for_2d,
                                                        generator.d_logits_y: batch_d_logits,
                                                        generator.z_u_y: batch_z_u,
                                                        generator.k: k,
                                                        generator.user_embeddings: alias_table.prob_users,
                                                        generator.item_embeddings: alias_table.prob_items})
            batch_item_embeddings = np.reshape(batch_item_embeddings, [-1])
            item_embeddings[index:index_end, k] = batch_item_embeddings
            index += batch_size
        Train_g_time += time.time() - time_start
    temp = np.max(item_embeddings, axis=0)
    temp = np.mean(temp) * config['lambda_y']
    time_start = time.time()
    item_embeddings = sess.run(generator.item_embedding_y_norm, {generator.item_embeddings: item_embeddings,
                                                                 generator.lambda_y: [temp]})
    Train_g_time += time.time() - time_start
    return item_embeddings


def train(config):
    # load data
    user_pos_train, user_pos_test = load_data(config)
    all_users = list(user_pos_train.keys())
    all_users.sort()
    user_num = config['user_num']
    item_num = config['item_num']

    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as fout:
        print(json.dumps(config), file=fout)
    train_log = open(os.path.join(config['output_dir'], 'train_log.txt'), 'w')

    # build model
    generator = GEN(config)
    discriminator = DIS(config)

    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    model_path = os.path.join(config['output_dir'], 'model/SD-GAR')
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config_tf)
    # init variable
    sess.run(tf.compat.v1.global_variables_initializer())

    # init embeddings
    print("%s; initializing embeddings and constructing alias table..." % get_current_time())
    prob_users, prob_items = get_init_embeddings(config)
    alias_table = AliasTable(prob_users, prob_items)
    sampler_d = SamplerD(config, alias_table=alias_table, generator=generator, sess=sess)
    print("%s; finished" % get_current_time())

    # minimax training
    best, best_gen = 0., 0.
    global Train_d_cnt, Gen_data_for_d_time, Train_d_time
    global Train_g_cnt, Gen_data_for_g_time, Train_g_time
    mretic_name = ['P@3', 'P@5', 'P@10', 'P@50', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@50', 'MRR']
    for epoch in range(76):
        # train discriminator
        if epoch > 0:
            Train_d_cnt += 1
            batch_num = 0
            # the loss incorporates four parts including total loss, positive sample loss, negtive sample loss and regularization
            loss_arr = np.array([0.] * 4)
            time_start = time.time()
            sampler_d.generate_data(user_pos_train, config['dis_sample_num'], shuffle=True)
            sampler_d.generate_neg_scores()
            Gen_data_for_d_time += time.time() - time_start
            data_len = len(sampler_d.data)
            time_start = time.time()
            index = 0
            while index < data_len:
                batch = sampler_d.get_next_batch(config['batch_size'])
                users = batch['users']
                pos_items = batch['pos_items']
                neg_items = batch['neg_items']
                neg_scores = batch['neg_scores']
                index += config['batch_size']
                _, batch_loss_list = sess.run(
                    [discriminator.get_update(), discriminator.get_loss()],
                    feed_dict={discriminator.u: users,
                               discriminator.pos_i: pos_items,
                               discriminator.neg_i: neg_items,
                               discriminator.g_scores: neg_scores})
                batch_loss_list[1] = np.mean(batch_loss_list[1])
                batch_loss_list[2] = np.mean(batch_loss_list[2])
                batch_loss_arr = np.array(batch_loss_list)
                loss_arr += batch_loss_arr
                batch_num += 1
            Train_d_time += time.time() - time_start
            loss_arr = loss_arr / batch_num
            curr_time = get_current_time()
            buf = "%s; epoch: %s; loss: %s; pos_loss: %s; neg_loss: %s; regular_loss: %s" % (
                curr_time, epoch, loss_arr[0], loss_arr[1], loss_arr[2], loss_arr[3])
            output_to_file(buf, train_log)

            if epoch % 5 == 0:
                result = eval(sess, discriminator, user_pos_train, user_pos_test)
                curr_time = get_current_time()
                buf = "\t%s; metrics:    \t%s" % (curr_time, '\t'.join(["%7s" % x for x in mretic_name]))
                output_to_file(buf, train_log)
                buf = "\t%s; performance:\t%s" % (curr_time, '\t'.join(["%.5f" % x for x in result]))
                output_to_file(buf, train_log)
                ndcg_50 = result[7]
                if ndcg_50 > best:
                    buf = '\tbest ndcg@50, saving the current model'
                    output_to_file(buf, train_log)
                    best = ndcg_50
                    saver.save(sess, model_path)
                    f_gen_embeddings = open(os.path.join(config['output_dir'], 'gen_embeddings.txt'), 'wb')
                    pickle.dump([alias_table.prob_users, alias_table.prob_items], f_gen_embeddings)

        if epoch % 5 == 0:
            Train_g_cnt += 1
            print("%s; computing partition function..." % get_current_time())
            z_u, d_logits = get_partition_funciton(sess, discriminator, generator, alias_table, config)

            # update user embeddings
            print("%s; computing u..." % get_current_time())
            prob_users = get_new_user_embeddings(sess, discriminator, generator, z_u, alias_table, config)
            print("%s; update alias table u..." % get_current_time())
            time_start = time.time()
            # update user alias table
            alias_table.update_users(prob_users)
            Train_g_time += time.time() - time_start
            print("%s; finish updating..." % get_current_time())

            print("%s; computing v..." % get_current_time())
            # update item embeddings
            prob_items = get_new_item_embeddings(sess, discriminator, generator, z_u, alias_table, config)

            print("%s; update alias table v..." % get_current_time())
            # update item alias table
            time_start = time.time()
            alias_table.update_items(prob_items)
            Train_g_time += time.time() - time_start
            print("%s; finish updating..." % get_current_time())

    output_to_file("cost on generating data for d: %s" % (Gen_data_for_d_time / Train_d_cnt), train_log)
    output_to_file("cost on training d: %s" % (Train_d_time / Train_d_cnt), train_log)
    output_to_file("cost on generating data for g: %s" % (Gen_data_for_g_time / Train_g_cnt), train_log)
    output_to_file("cost on training g: %s" % (Train_g_time / Train_g_cnt), train_log)
    train_log.close()


def run(conf_name):
    pass
    from config import conf
    if conf_name not in conf:
        raise Exception('config name not in the config.py')
    config = conf[conf_name]
    train(config)


if __name__ == '__main__':
    pass
    run(sys.argv[1])
