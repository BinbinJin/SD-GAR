import tensorflow as tf
import numpy as np
from utils import *


class DIS:
    def __init__(self, config):
        self.itemNum = config["item_num"]
        self.userNum = config["user_num"]
        self.emb_dim = config["dis_emb_dim"]
        self.neg_sample_num = config['dis_sample_num']
        self.lamda = config['dis_lambda'] / config['batch_size']
        self.learning_rate = config['learning_rate']
        self.d_params = []
        self.d_summary = []

        with tf.compat.v1.variable_scope("MF", reuse=tf.compat.v1.AUTO_REUSE):
            self.user_embeddings = tf.compat.v1.get_variable('user_embeddings', [self.userNum, self.emb_dim],
                                                             dtype=tf.float32,
                                                             initializer=tf.compat.v1.keras.initializers.glorot_uniform())
            self.item_embeddings = tf.compat.v1.get_variable('item_embeddings', [self.itemNum, self.emb_dim],
                                                             dtype=tf.float32,
                                                             initializer=tf.compat.v1.keras.initializers.glorot_uniform())
            self.item_bias = tf.compat.v1.get_variable('item_bias', [self.itemNum], dtype=tf.float32,
                                                       initializer=tf.zeros_initializer)
            self.d_params += [self.user_embeddings, self.item_embeddings, self.item_bias]
            self.d_summary.append(tf.compat.v1.summary.histogram('user_embedding', self.user_embeddings))
            self.d_summary.append(tf.compat.v1.summary.histogram('item_embedding', self.item_embeddings))
            self.d_summary.append(tf.compat.v1.summary.histogram('item_bias', self.item_bias))

            self.u = tf.compat.v1.placeholder(tf.int32, [None], 'dis_user')
            self.pos_i = tf.compat.v1.placeholder(tf.int32, [None], 'dis_pos_items')
            self.neg_i = tf.compat.v1.placeholder(tf.int32, [None, self.neg_sample_num], 'dis_neg_items')
            self.g_scores = tf.compat.v1.placeholder(tf.float32, [None, self.neg_sample_num], 'dis_gen_scores')

            self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
            self.pos_i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos_i)
            self.pos_bias = tf.gather(self.item_bias, self.pos_i)
            self.neg_i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg_i)
            self.neg_bias = tf.gather(self.item_bias, self.neg_i)

            u_embedding_ext = tf.expand_dims(self.u_embedding, axis=1)

            # dot part
            self.pos_logits = self._build_dot_product(self.u_embedding, self.pos_i_embedding, self.pos_bias)
            self.pos_logits = tf.reshape(self.pos_logits, [-1, 1])
            self.neg_logits = self._build_dot_product(u_embedding_ext, self.neg_i_embedding, self.neg_bias)

            self.pos_loss = -tf.math.log_sigmoid(self.pos_logits)
            T = config['T']
            self.neg_weight = tf.nn.softmax(tf.nn.softplus(self.neg_logits) / T - tf.math.log(self.g_scores))
            self.neg_loss = tf.nn.softplus(self.neg_logits)
            self.neg_w_loss = tf.reduce_sum(tf.multiply(self.neg_weight, self.neg_loss), axis=1, keepdims=True)
            self.data_loss = tf.reduce_mean(tf.concat([self.pos_loss, self.neg_w_loss], axis=0))

            self.regular_loss = tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.pos_i_embedding) + tf.nn.l2_loss(
                self.neg_i_embedding) / self.neg_sample_num + tf.nn.l2_loss(self.pos_bias) + tf.nn.l2_loss(
                self.neg_bias) / self.neg_sample_num
            self.regular_loss = self.lamda * self.regular_loss
            self.total_loss = self.data_loss + self.regular_loss
            self.d_summary.append(tf.compat.v1.summary.scalar('dis_loss', self.total_loss))

            g_opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            self.gan_updates = g_opt.minimize(self.total_loss, var_list=self.d_params)
            self.summary_merge = tf.compat.v1.summary.merge(self.d_summary)

            self.all_ratings = tf.matmul(self.u_embedding, self.item_embeddings, transpose_b=True)
            self.all_ratings = self.all_ratings + tf.reshape(self.item_bias, [1, self.itemNum])

    def get_update(self):
        return self.gan_updates

    def get_summary(self):
        return self.summary_merge

    def get_neg_logits(self):
        return self.neg_logits

    def get_pos_logits(self):
        return self.pos_logits

    def get_loss(self):
        return [self.total_loss, self.pos_loss, self.neg_w_loss, self.regular_loss]

    def _build_dot_product(self, inputs_users, inputs_items, inputs_bias):
        dot_product = tf.reduce_sum(tf.multiply(inputs_users, inputs_items), axis=-1) + inputs_bias
        return dot_product

    def get_all_ratings(self):
        return self.all_ratings

    def get_user_embeddings(self):
        return self.user_embeddings

    def get_item_embeddings(self):
        return self.item_embeddings

    def get_item_bias(self):
        return self.item_bias
