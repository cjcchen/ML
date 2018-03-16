import time
import os
import numpy as np
import tensorflow as tf
import resnet 
from dataset.imagenet import imagenet_input
from component import network_train

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/img_train/', 'data dir.')
tf.app.flags.DEFINE_string('test_data_dir', 'data/img_train/', 'data dir.')
tf.app.flags.DEFINE_string('meta_data_dir', 'data/img_train/', 'data dir.')
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_string('cpu_mode', 'cpu', "cpu? or gpu?")

def main(argv=None):  

    img_size = 224
    num_class = 100
    [x,y],words=imagenet_input.image_input(FLAGS.meta_data_dir, FLAGS.test_data_dir, batch_size=FLAGS.batch_size, mode='eval')
    #[test_x,test_y],words=imagenet_input.image_input(FLAGS.meta_data_dir, FLAGS.test_data_dir, batch_size=FLAGS.batch_size, mode='eval')

    with tf.device("/"+FLAGS.cpu_mode+":0"):
        train_x=tf.placeholder(tf.float32, [None,img_size,img_size,3])
        train_y=tf.placeholder(tf.int32, [None]) 
        #model = resnet_model.ResNet('train')
        with tf.variable_scope("resnet", reuse=None):
            model = resnet.ResNet(train_x,train_y,num_class=num_class, mode='train')

        #with tf.variable_scope("resnet", reuse=True):
        #    eval_model = resnet.ResNet(train_x,train_y,num_class=num_class, mode='eval')
    network_train.train(model,model, x,y,x,y)
    #network_train.train(model,eval_model, x,y,test_x,test_y)

if __name__ == '__main__':
    tf.app.run()
