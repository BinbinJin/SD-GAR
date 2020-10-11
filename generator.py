import numpy as np
import tensorflow as tf


class GEN:
    def __init__(self, config):
        self.itemNum = config['item_num']
        self.userNum = config['user_num']
        self.emb_dim = config['gen_emb_dim']
        self.sample_num_item = config['gen_sample_num_item']
        self.sample_num_user = config['gen_sample_num_user']

        self.user_embeddings = tf.compat.v1.placeholder(tf.float32, [self.userNum, self.emb_dim], 'user_embeddings')
        self.item_embeddings = tf.compat.v1.placeholder(tf.float32, [self.itemNum, self.emb_dim], 'item_embeddings')
        user_embeddings = self.user_embeddings * 1.0
        item_embeddings = self.item_embeddings * 1.0

        self.u_x = tf.compat.v1.placeholder(tf.int32, [None], 'user_x')
        self.i_x = tf.compat.v1.placeholder(tf.int32, [None, self.sample_num_item], 'item_x')
        self.d_logits_x = tf.compat.v1.placeholder(tf.float32, [None, self.sample_num_item], 'd_logits_x')
        # self.z_u_x is the partition function which is pre-calculated before updating the user embeddings
        self.z_u_x = tf.compat.v1.placeholder(tf.float32, [None], 'z_u_x')
        self.i_d = tf.compat.v1.placeholder(tf.int32, [None], 'item_d')

        self.u_embeddings_x = tf.nn.embedding_lookup(user_embeddings, self.u_x)
        self.i_embeddings_x = tf.nn.embedding_lookup(item_embeddings, self.i_x)
        self.i_embeddings_d = tf.nn.embedding_lookup(item_embeddings, self.i_d)

        # logits for discriminator
        self.logits_d = tf.reduce_sum(tf.multiply(self.u_embeddings_x, self.i_embeddings_d), axis=1)

        # logits for generator
        user_embeddings_ext = tf.expand_dims(self.u_embeddings_x, axis=1)
        self.logits_x = tf.reduce_sum(tf.multiply(user_embeddings_ext, self.i_embeddings_x), axis=2)
        self.log_logits_x = tf.math.log(self.logits_x)

        # d_logits_x: logits from discriminator when updating user embeddings
        self.log_d_logits_x = tf.nn.softplus(self.d_logits_x)
        T = config['T']
        d_logits_x = tf.nn.softplus(self.d_logits_x) / T

        # compute partition function
        self.partition = tf.reduce_mean(tf.exp(d_logits_x - self.log_logits_x), axis=1, keepdims=True)

        # update user embeddings and leading to unnormalized embeddings self.v
        self.i_emb_gather = tf.gather(item_embeddings, self.i_x)
        self.importance_x = tf.exp(d_logits_x - self.log_logits_x) / tf.expand_dims(self.z_u_x, axis=1)
        self.weight_x = tf.multiply(self.importance_x, self.log_d_logits_x)
        self.weight_x = tf.expand_dims(tf.math.divide(self.weight_x, self.logits_x), axis=2)
        self.v = tf.reduce_mean(tf.multiply(self.i_emb_gather, self.weight_x), axis=1)

        # normalize user embedding with temperature lambda_x
        self.lambda_x = tf.compat.v1.placeholder(tf.float32, None, 'lambda_x')
        self.user_embeddings_norm = tf.nn.softmax(tf.math.divide(self.user_embeddings, self.lambda_x))

        # get all ratings
        u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.u_x)
        self.all_ratings = tf.matmul(u_embeddings, self.item_embeddings, transpose_b=True)

        # update item embedding
        self.k = tf.compat.v1.placeholder(tf.int32)
        self.u_y = tf.compat.v1.placeholder(tf.int32, [None, self.sample_num_user], 'user_y')
        self.i_y = tf.compat.v1.placeholder(tf.int32, [None], 'item_y')
        self.d_logits_y = tf.compat.v1.placeholder(tf.float32, [None, self.sample_num_user], 'd_logits_y')
        # z_u_y is the partition function which is pre-calculated before updating the item embeddings
        self.z_u_y = tf.compat.v1.placeholder(tf.float32, [None, self.sample_num_user], 'z_u_y')
        self.u_embeddings_y = tf.nn.embedding_lookup(user_embeddings, self.u_y)
        self.i_embeddings_y = tf.nn.embedding_lookup(item_embeddings, self.i_y)

        item_embeddings_ext = tf.expand_dims(self.i_embeddings_y, axis=1)
        self.logits_y = tf.reduce_sum(tf.multiply(item_embeddings_ext, self.u_embeddings_y), axis=2)
        self.log_logits_y = tf.math.log(self.logits_y)

        # d_logits_y: logits from discriminator when updating item embeddings
        self.log_d_logits_y = tf.nn.softplus(self.d_logits_y)
        T = config['T']
        d_logits_y = tf.nn.softplus(self.d_logits_y) / T

        # updating item embeddings and leading to unnormalized embeddings self.s
        self.importance_y = tf.exp(d_logits_y - self.log_logits_y) / self.z_u_y
        self.weight_y = tf.multiply(self.importance_y, self.log_d_logits_y)
        self.u_hat = tf.reduce_sum(self.user_embeddings, axis=0, keepdims=True)
        self.weight_y = tf.reduce_mean(self.weight_y, axis=1, keepdims=True)
        self.s = tf.multiply(self.u_hat, self.weight_y)
        self.s = tf.gather(self.s, self.k, axis=1)

        # normalize item embedding with temperature lambda_y
        self.lambda_y = tf.compat.v1.placeholder(tf.float32, [1], name='lambda_y')
        self.item_embedding_y_norm = tf.nn.softmax(tf.math.divide(self.item_embeddings, self.lambda_y), axis=0)

        # generate random number with gpu
        self.random_k = tf.cast(
            tf.random.uniform([self.userNum, self.sample_num_item], 0, self.emb_dim, tf.float32), tf.int32)
        self.random_j = tf.cast(
            tf.random.uniform([self.userNum, self.sample_num_item], 0, self.itemNum, tf.float32), tf.int32)
        self.random_prob_j = tf.random.uniform([self.userNum, self.sample_num_item * 2], dtype=tf.float32)

        self.random_u = tf.cast(
            tf.random.uniform([self.itemNum, self.sample_num_user], 0, self.userNum, tf.float32), tf.int32)
        self.random_prob_u = tf.random.uniform([self.itemNum, self.sample_num_user], dtype=tf.float32)

        # sample users with gpu
        self.sample_u = self._sample_user()

    def get_logits_for_d(self):
        return self.logits_d

    def get_partition(self):
        return self.partition

    def get_new_user_embeddings(self):
        return self.v

    def get_new_item_embeddings(self):
        return self.s

    def get_all_ratings(self):
        return self.all_ratings

    def get_random_for_item(self):
        return [self.random_k, self.random_j, self.random_prob_j]

    def get_random_for_user(self):
        return [self.random_u, self.random_prob_u]

    def _sample_user(self):
        self.threshold_user = tf.compat.v1.placeholder(tf.float32, [1, self.userNum])
        self.index_1_user = tf.compat.v1.placeholder(tf.int32, [1, self.userNum])
        self.index_2_user = tf.compat.v1.placeholder(tf.int32, [1, self.userNum])
        random_user = tf.cast(tf.random.uniform([1, self.itemNum * self.sample_num_user], 0, self.userNum, tf.float32),
                              tf.int32)
        random_prob = tf.random.uniform([1, self.itemNum * self.sample_num_user], dtype=tf.float32)
        sample_user = self._alias_table_sample(self.threshold_user,
                                               self.index_1_user,
                                               self.index_2_user,
                                               random_user,
                                               random_prob)
        sample_u = tf.reshape(sample_user, [self.itemNum, self.sample_num_user])
        return sample_u

    def _alias_table_sample(self, threshold, index_1, index_2, random_idx, random_prob):
        x_dim = threshold.shape.as_list()[0]
        sample_num = random_idx.shape.as_list()[1]
        x_cor = tf.matmul(tf.linalg.tensor_diag(tf.range(x_dim)), tf.ones_like(random_idx, tf.int32))
        x_cor_1d = tf.reshape(x_cor, [-1, 1])
        y_cor_1d = tf.reshape(random_idx, [-1, 1])
        cor = tf.concat([x_cor_1d, y_cor_1d], axis=1)
        sample_threshold = tf.reshape(tf.gather_nd(threshold, cor), [-1, sample_num])
        sample_idx_1 = tf.reshape(tf.gather_nd(index_1, cor), [-1, sample_num])
        sample_idx_2 = tf.reshape(tf.gather_nd(index_2, cor), [-1, sample_num])
        sample_idx = tf.where(tf.less_equal(random_prob, sample_threshold), sample_idx_1, sample_idx_2)
        return sample_idx
