from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from dataset.imagenet import imagenet_input
import zfnet
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cpu_mode', 'cpu', "cpu? or gpu?")
tf.app.flags.DEFINE_string('data_dir', '', "data dir")
tf.app.flags.DEFINE_string('test_data_dir', '', "data dir")
tf.app.flags.DEFINE_string('meta_data_dir', '', "meta_data dir")
tf.app.flags.DEFINE_integer('batch_size', 5, "batch size")
tf.app.flags.DEFINE_string('save_dir', "checkpoints", "batch size")
tf.app.flags.DEFINE_string('log_dir', "logs", "batch size")
tf.app.flags.DEFINE_float('lr', "0.01", "batch size")

def data_func(session,model):
    global x,y
    m_x,m_y = session.run([x,y])
    feed_dict_train = {model.x: m_x, model.y:m_y}
    return feed_dict_train

def main(argv=None):  
    global x,y
    [x,y],words=imagenet_input.image_input(FLAGS.meta_data_dir, FLAGS.data_dir, batch_size=FLAGS.batch_size, mode='train')
    #[test_x,test_y],words=imagenet_input.image_input(FLAGS.meta_data_dir,FLAGS.test_data_dir, batch_size=FLAGS.batch_size,mode='eval')

    with tf.device("/"+FLAGS.cpu_mode+":0"):
        train_x=tf.placeholder(tf.float32, [None,224,224,3])
        train_y=tf.placeholder(tf.int32, [None]) 
        with tf.variable_scope("zfnet", reuse=None):
            model = zfnet.ZFNet(train_x,train_y,num_class=1000, mode='train', pre_fix="zfnet")
            model.set_para(FLAGS.save_dir, FLAGS.log_dir, FLAGS.lr)
        #with tf.variable_scope("zfnet", reuse=True):
        #    eval_model = zfnet.ZFNet(train_x,train_y,num_class=1000, mode='eval')

    with tf.variable_scope("zfnet", reuse=tf.AUTO_REUSE):
        model.train(data_func)

if __name__ == '__main__':
    tf.app.run()
