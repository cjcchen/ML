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
tf.app.flags.DEFINE_string('test_data_dir', '', "data dir")
tf.app.flags.DEFINE_string('meta_data_dir', '', "meta_data dir")
tf.app.flags.DEFINE_integer('batch_size', 5, "batch size")

def main(argv=None):  

    [x,y],words=imagenet_input.image_input(FLAGS.meta_data_dir, FLAGS.data_dir, batch_size=FLAGS.batch_size, mode='train')
    [test_x,test_y],words=imagenet_input.image_input(FLAGS.meta_data_dir,FLAGS.test_data_dir, batch_size=FLAGS.batch_size,mode='eval')

    with tf.device("/"+FLAGS.cpu_mode+":0"):
        train_x=tf.placeholder(tf.float32, [None,224,224,3])
        train_y=tf.placeholder(tf.int32, [None]) 
        #model = resnet_model.ResNet('train')
        with tf.variable_scope("zfnet", reuse=None):
            model = zfnet.ZFNet(train_x,train_y,num_class=1000, mode='train')

        with tf.variable_scope("zfnet", reuse=True):
            eval_model = zfnet.ZFNet(train_x,train_y,num_class=1000, mode='eval')

    #with tf.Session() as sess:
    #      sess.run(tf.initialize_all_variables())
    #      threads = tf.train.start_queue_runners(sess)
    #      d,label=sess.run([x,y])
    #      print d.shape, label.shape, label
    network_train.train(model,eval_model, x,y,test_x,test_y)

if __name__ == '__main__':
    tf.app.run()
