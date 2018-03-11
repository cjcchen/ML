from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from dataset.imagenet import imagenet_input
from component import network_train
import zfnet
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cpu_mode', 'cpu', "cpu? or gpu?")
tf.app.flags.DEFINE_string('data_dir', '', "data dir")
tf.app.flags.DEFINE_integer('batch_size', 5, "batch size")

def main(argv=None):  

    x,y=imagenet_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size)
    test_x,test_y=imagenet_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size,data_type='test')

    with tf.device("/"+FLAGS.cpu_mode+":0"):
        #model = resnet_model.ResNet('train')
        with tf.variable_scope("zfnet", reuse=None):
            train_x=tf.placeholder(tf.float32, [None,224,224,3])
            train_y=tf.placeholder(tf.int32, [None]) 
            model = zfnet.ZFNet(train_x,train_y,num_class=1000, mode='train')

        with tf.variable_scope("zfnet", reuse=True):
            e_x=tf.placeholder(tf.float32, [None,224,224,3])
            e_y=tf.placeholder(tf.int32, [None]) 
            eval_model = zfnet.ZFNet(e_x,e_y,num_class=1000, mode='eval')

    network_train.train(model,eval_model, x,y,test_x,test_y)

if __name__ == '__main__':
    tf.app.run()
