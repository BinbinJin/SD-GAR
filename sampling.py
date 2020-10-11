import numpy as np
import random
import tensorflow as tf
import os
import multiprocessing
from alias_method import AliasTable
from utils import timer, load_matrix_from_file, get_current_time


class SamplerD:
    def __init__(self, config, **kwargs):
        """
        sampler for discriminator
        :param config:
        :param kwargs:
        """
        self.config = config
        self.alias_table = kwargs['alias_table']
        self.sess = kwargs['sess']
        self.generator = kwargs['generator']
        self.data = None
        self.data_index = 0
        self.neg_scores = None
        self.sample_num = 0
        self.batch_size = 10240

    def generate_data(self, user_pos_train, sample_num, shuffle=False):
        self.data = self._generate_data(user_pos_train, sample_num, shuffle)
        self.data_index = 0
        self.sample_num = sample_num

    def reset_index(self):
        self.data_index = 0

    def _generate_data(self, user_pos_train, sample_num_per_pos, shuffle):
        data = []
        for u in user_pos_train:
            pos = user_pos_train[u]
            pos_set = set(pos)
            negs, negs_k = [], []
            sample_num = len(pos) * sample_num_per_pos
            while len(negs) < sample_num:
                samples, inter_var = self.alias_table.sample_for_u(u, sample_num - len(negs))
                for j, k in zip(samples, inter_var):
                    if j in pos_set:
                        continue
                    negs.append(j)
                    negs_k.append(k)
            for i in range(len(pos)):
                neg = negs[i * sample_num_per_pos:(i + 1) * sample_num_per_pos]
                neg_k = negs_k[i * sample_num_per_pos:(i + 1) * sample_num_per_pos]
                data.append((u, pos[i], neg, neg_k))
        if shuffle:
            random.shuffle(data)
        return data

    def generate_neg_scores(self):
        self._generate_neg_scores()

    def _generate_neg_scores(self):
        batch_size = self.batch_size
        sample_num = self.sample_num
        index = 0
        sess = self.sess
        generator = self.generator
        scores = np.zeros([len(self.data), sample_num], np.float)
        while index < len(self.data):
            index_end = min(len(self.data), index + batch_size)
            batch_users, batch_items = [], []
            for idx in range(index, index_end):
                user, _, negs = self.data[idx][:3]
                batch_users += [user] * sample_num
                batch_items += list(negs)
            batch_users = np.array(batch_users)
            batch_items = np.array(batch_items)

            batch_scores = sess.run(generator.get_logits_for_d(),
                                    feed_dict={generator.u_x: batch_users,
                                               generator.i_d: batch_items,
                                               generator.item_embeddings: self.alias_table.prob_items,
                                               generator.user_embeddings: self.alias_table.prob_users})
            batch_scores = np.reshape(batch_scores, [-1, sample_num])
            scores[index:index_end] = batch_scores
            index += batch_size
        self.neg_scores = scores

    def get_next_batch(self, batch_size):
        index = self.data_index
        index_end = min(len(self.data), index + batch_size)
        batch_users, batch_pos, batch_negs, batch_negs_scores = [], [], [], []
        for idx in range(index, index_end):
            user, pos, negs = self.data[idx][:3]
            batch_users.append(user)
            batch_pos.append(pos)
            batch_negs.append(negs)
            batch_negs_scores.append(self.neg_scores[idx])
        self.data_index += batch_size
        batch = {
            'users': batch_users,
            'pos_items': batch_pos,
            'neg_items': batch_negs,
            'neg_scores': batch_negs_scores
        }
        return batch


class SamplerG:
    def __init__(self, config, sess, discriminator, alias_table, tag):
        """
        sampler for generator
        :param config:
        :param sess:
        :param discriminator:
        :param alias_table:
        :param tag: 'x' for user, 'y' for item
        """
        if tag != 'x' and tag != 'y':
            raise Exception('wrong tag!')
        self.config = config
        self.sess = sess
        self.discriminator = discriminator
        self.alias_table = alias_table
        self.tag = tag
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.emb_dim = config['gen_emb_dim']
        self.data_user = None
        self.data_item = None
        self.d_logits = None
        self.index = 0
        self.batch_size = 10240

    def generate_data(self, **kwargs):
        sample_num = kwargs['sample_num']
        random_dic = kwargs['random_dic'] if 'random_dic' in kwargs else None
        self._generate_data(sample_num, random_dic)
        self.index = 0

    def _generate_data(self, sample_num, random_dic=None):
        data = []
        for u in range(self.user_num):
            random_dic_each_user = None
            if random_dic is not None:
                random_dic_each_user = {'random_k': random_dic['random_k'][u],
                                        'random_j': random_dic['random_j'][u],
                                        'random_prob': random_dic['random_prob'][u]}
            data.append(self.alias_table.sample_for_u(u, sample_num, random_dic_each_user)[0])
        self.data_item = data
        self.data_user = list(range(self.user_num))

    def generate_logits(self, **kwargs):
        self._generate_logits()

    def _generate_logits(self):
        if self.tag == 'x':
            x_dim = len(self.data_item)
            y_dim = len(self.data_item[0])
        else:
            x_dim = len(self.data_user)
            y_dim = len(self.data_user[0])
        self.d_logits = np.zeros([x_dim, y_dim], np.float)
        self.reset_index()
        batch_size = self.batch_size
        index = 0
        while index < x_dim:
            batch = self.get_next_batch(batch_size)
            index_end = min(x_dim, index + batch_size)
            batch_users_for_1d = batch['batch_users_for_1d']
            batch_items_for_1d = batch['batch_items_for_1d']
            batch_logits = self.sess.run(self.discriminator.get_pos_logits(),
                                         {self.discriminator.u: batch_users_for_1d,
                                          self.discriminator.pos_i: batch_items_for_1d})
            batch_logits = np.reshape(batch_logits, [-1, y_dim])
            self.d_logits[index:index_end] = batch_logits
            index += batch_size
        self.reset_index()

    def reset_index(self):
        self.index = 0

    def get_next_batch(self, batch_size):
        index = self.index
        index_end = min(index + batch_size, len(self.data_user))
        users = self.data_user[index:index_end]
        items = self.data_item[index:index_end]
        batch_d_logits = self.d_logits[index:index_end]
        self.index += batch_size
        batch_users_for_2d = np.array(users)
        batch_items_for_2d = np.array(items)
        if len(np.shape(batch_users_for_2d)) == 1:
            rep_num = np.shape(batch_items_for_2d)[1]
            batch_users_for_1d = [[i] * rep_num for i in users]
            batch_users_for_1d = np.reshape(np.array(batch_users_for_1d), [-1])
            batch_items_for_1d = np.reshape(batch_items_for_2d, [-1])
        else:
            rep_num = np.shape(batch_users_for_2d)[1]
            batch_items_for_1d = [[i] * rep_num for i in items]
            batch_items_for_1d = np.reshape(np.array(batch_items_for_1d), [-1])
            batch_users_for_1d = np.reshape(batch_users_for_2d, [-1])
        batch = {
            'batch_users_for_2d': batch_users_for_2d,
            'batch_items_for_2d': batch_items_for_2d,
            'batch_users_for_1d': batch_users_for_1d,
            'batch_items_for_1d': batch_items_for_1d,
            'batch_d_logits': batch_d_logits
        }
        return batch
