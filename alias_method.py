import queue
import math
import numpy as np


class AliasTable:
    def __init__(self, prob_users, prob_items, AT_update=True):
        self.prob_users = prob_users
        self.prob_items = prob_items
        self.user_num = len(prob_users)
        self.emb_dim = len(prob_users[0])
        self.item_num = len(prob_items)
        self.k_u_alias_tables, self.j_k_alias_tables = None, None

        if AT_update:
            self.update_users(self.prob_users)
            self.update_items(self.prob_items)

    def update_users(self, prob_users):
        self.prob_users = prob_users
        k_u_alias_tables = []
        for u in range(self.user_num):
            k_u_alias_tables.append(self.alias_table_construct(prob_users[u]))
        self.k_u_alias_tables = k_u_alias_tables

    def update_items(self, prob_items):
        self.prob_items = prob_items
        j_k_alias_tables = []
        for k in range(self.emb_dim):
            j_k_alias_tables.append(self.alias_table_construct(prob_items[:, k]))
        self.j_k_alias_tables = j_k_alias_tables

    def sample_for_u(self, user, sample_num, random_dic=None):
        if random_dic is not None:
            rand_k_arr = random_dic['random_k']
            rand_j_arr = random_dic['random_j']
            rand_prob_arr = random_dic['random_prob']
        else:
            rand_k_arr = np.random.randint(0, self.emb_dim, sample_num)
            rand_j_arr = np.random.randint(0, self.item_num, sample_num)
            rand_prob_arr = np.random.rand(sample_num * 2)
        samples, intermediate_var = [], []
        for i in range(sample_num):
            idx = rand_k_arr[i]
            prob = rand_prob_arr[2 * i]
            t = idx if prob <= self.k_u_alias_tables[user][idx][0] else self.k_u_alias_tables[user][idx][1]

            idx = rand_j_arr[i]
            prob = rand_prob_arr[2 * i + 1]
            j = idx if prob <= self.j_k_alias_tables[t][idx][0] else self.j_k_alias_tables[t][idx][1]

            intermediate_var.append(t)
            samples.append(j)
        return samples, intermediate_var

    @staticmethod
    def alias_table_construct(multinomial):
        size = len(multinomial)
        table = np.zeros_like(multinomial)
        alias_table_list = [None] * size
        for i in range(len(multinomial)):
            table[i] = multinomial[i] * size

        index = list(range(size))
        quene_small = queue.Queue()
        queue_large = queue.Queue()
        for item in zip(table, index):
            if item[0] < 1.0:
                quene_small.put(item)
            elif item[0] > 1.0:
                queue_large.put(item)
            else:
                alias_table_list[item[1]] = (1.0, None)

        small = quene_small.get() if not quene_small.empty() else None
        large = queue_large.get() if not queue_large.empty() else None
        while small is not None and large is not None:
            complete = (small[0], large[1])
            alias_table_list[small[1]] = complete
            rest_prob = small[0] + large[0] - 1.0
            if rest_prob >= 1.0:
                small = quene_small.get() if not quene_small.empty() else None
                large = (rest_prob, large[1])
            else:
                small = (rest_prob, large[1])
                large = queue_large.get() if not queue_large.empty() else None

        while small is not None:
            alias_table_list[small[1]] = (1.0, None)
            small = quene_small.get() if not quene_small.empty() else None

        while large is not None:
            alias_table_list[large[1]] = (1.0, None)
            large = queue_large.get() if not queue_large.empty() else None

        return alias_table_list

    @staticmethod
    def sample_from_alias_table_1d(alias_table, sample_num, random_dic):
        num = len(alias_table)
        if random_dic is not None:
            rand_idx_arr = random_dic['random_u']
            rand_prob_arr = random_dic['random_prob']
        else:
            rand_idx_arr = np.random.randint(0, num, sample_num)
            rand_prob_arr = np.random.rand(sample_num)
        samples = []
        for i in range(sample_num):
            idx = rand_idx_arr[i]
            prob = rand_prob_arr[i]
            idx = idx if prob <= alias_table[idx][0] else alias_table[idx][1]
            samples.append(idx)
        return samples


if __name__ == '__main__':
    prob = np.array(
        [0.062499988824129105, 0.062499988824129105, 0.0625000074505806, 0.062499988824129105, 0.0625000074505806,
         0.0625000074505806, 0.0625000074505806, 0.0625000074505806, 0.0625000074505806, 0.062499988824129105,
         0.062499988824129105, 0.0625000074505806, 0.0625000074505806, 0.0625000074505806, 0.0625000074505806,
         0.06250003725290298])
    AliasTable.alias_table_construct(prob)
