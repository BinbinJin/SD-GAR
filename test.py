import os
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from utils import load_data, get_current_time
from evaluation import eval
from config import conf
from discriminator import DIS


def test(conf_name):
    if conf_name not in conf:
        raise Exception("config name not in config.py")
    config = conf[conf_name]
    print("load data...")
    user_pos_train, user_pos_test = load_data(config)
    all_users = list(user_pos_train.keys())
    all_users.sort()

    print("load model...")
    discriminator = DIS(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config_tf)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    checkpoint = os.path.join(config['output_dir'], 'model/SD-GAR')
    saver.restore(sess, checkpoint)

    mretic_name = ['p@3', 'P@5', 'P@10', 'P@50', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@50', 'MRR']
    print("%s" % get_current_time(), '\t'.join(["%7s" % x for x in mretic_name]))
    res = eval(sess, discriminator, user_pos_train, user_pos_test)
    print("%s" % get_current_time(), '\t'.join(["%.5f" % x for x in res]))


if __name__ == '__main__':
    conf_name = sys.argv[1]
    test(conf_name)
