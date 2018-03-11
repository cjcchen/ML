import time
import os
import numpy as np
import tensorflow as tf
import resnet 
from dataset.imagenet import imagenet_input
from component import network_train

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/img_train/', 'data dir.')
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_string('cpu_mode', 'cpu', "cpu? or gpu?")

def main(argv=None):  

    img_size = 224
    num_class = 100
    x,y=imagenet_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size)
    test_x,test_y=imagenet_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size,data_type='test')

    with tf.device("/"+FLAGS.cpu_mode+":0"):
        #model = resnet_model.ResNet('train')
        with tf.variable_scope("resnet", reuse=None):
            train_x=tf.placeholder(tf.float32, [None,img_size,img_size,3])
            train_y=tf.placeholder(tf.int32, [None]) 
            model = resnet.ResNet(train_x,train_y,num_class=num_class, mode='train')

        with tf.variable_scope("resnet", reuse=True):
            e_x=tf.placeholder(tf.float32, [None,img_size,img_size,3])
            e_y=tf.placeholder(tf.int32, [None]) 
            eval_model = resnet.ResNet(e_x,e_y,num_class=num_class, mode='eval')
    network_train.train(model,eval_model, x,y,test_x,test_y)

if __name__ == '__main__':
    tf.app.run()
