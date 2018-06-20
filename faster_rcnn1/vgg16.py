import inspect
import os

import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.slim as slim

VGG_MEAN = [103.939, 116.779, 123.68]

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class Vgg16:
    def __init__(self):
        self._variables_to_fix={}
        self._scope = "vgg_16"
        pass
    def _image_to_head(self, image, is_training, reuse=None):
        net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                            trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                            trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                            trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv5')
        self.conv5_3=net
        return net

    def get_output(self):
        return self.conv5_3

    '''
    def get_variables_to_restore(self, variables, var_keep_dic):
        print ("keep dic:",var_keep_dic)
        print ("var:",variables)
        variables_to_restore = []
        for v in variables:
          # exclude the first conv layer to swap RGB to BGR
          if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
            self._variables_to_fix[v.name] = v
            continue
          if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
          with tf.device("/cpu:0"):
            # fix the vgg16 issue from conv weights to fc weights
            # fix RGB to BGR
            conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
            restorer_fc = tf.train.Saver({self._scope + "/conv1/conv1_1/weights": conv1_rgb})
            restorer_fc.restore(sess, pretrained_model)

            sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                tf.reverse(conv1_rgb, [2])))
    '''
    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
          # exclude the conv weights that are fc weights in vgg16
          if v.name == (self._scope + '/fc6/weights:0') or \
             v.name == (self._scope + '/fc7/weights:0'):
            self._variables_to_fix[v.name] = v
            print ("fix var",v.name,v.shape)
            continue
          # exclude the first conv layer to swap RGB to BGR
          if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
            self._variables_to_fix[v.name] = v
            continue
          if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)

        print ("var:",variables)
        print ("keep:",var_keep_dic)
        print ("get restore:",variables_to_restore)
        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
          with tf.device("/cpu:0"):
            # fix the vgg16 issue from conv weights to fc weights
            # fix RGB to BGR
            fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
            fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
            conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
            restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                          self._scope + "/fc7/weights": fc7_conv,
                                          self._scope + "/conv1/conv1_1/weights": conv1_rgb})
            restorer_fc.restore(sess, pretrained_model)

            sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                                self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
            sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                                self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
            sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                tf.reverse(conv1_rgb, [2])))

    def build(self, rgb, is_training = True):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        with tf.variable_scope("vgg_16"):
            return self._image_to_head(rgb,is_training)
        start_time = time.time()
        with tf.variable_scope("conv1"):
            self.conv1_1 = self.conv_layer(rgb, 64,  False, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, 64, False, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        with tf.variable_scope("conv2"):
            self.conv2_1 = self.conv_layer(self.pool1, 128, False, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, 128, False, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        with tf.variable_scope("conv3"):
            self.conv3_1 = self.conv_layer(self.pool2, 256, is_training, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, 256, is_training, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, 256, is_training, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        with tf.variable_scope("conv4"):
            self.conv4_1 = self.conv_layer(self.pool3, 512, is_training, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, 512, is_training, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, 512, is_training, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        with tf.variable_scope("conv5"):
            self.conv5_1 = self.conv_layer(self.pool4, 512, is_training, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, 512, is_training, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, 512, is_training, "conv5_3")

        print("build model finished: %ds" % (time.time() - start_time))

    def get_output(self):
        return self.conv5_3

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, x, output_channel, is_training, name):
        with tf.variable_scope(name):
            xshape = [3, 3, x.get_shape()[-1], output_channel]
            initializer = tf.contrib.layers.xavier_initializer( uniform=False )
            weights = tf.get_variable('weights', shape=xshape, dtype='float', initializer=initializer, trainable=is_training)

            conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0), trainable=is_training)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, input, output_dim, name):
        with tf.variable_scope(name):
            w = tf.get_variable('weights', [input.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training)
            o = tf.matmul(input, w)

            if bias:
                b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0), trainable=is_training)
                o = o + b

            if func:
                o = func(o)

            return o

def restore(model, checkpoint_dir):

    print ("var:",tf.trainable_variables())
    config_proto = tf.ConfigProto()
    sess = tf.Session(config=config_proto)
    saver = tf.train.Saver()
    try:
        print("Trying to restore last checkpoint ...:",checkpoint_dir)
        saver.restore(sess, save_path=checkpoint_dir)
        print("restore last checkpoint %s done"%checkpoint_dir)
    except Exception as e:
        print("Failed to restore checkpoint. Initializing variables instead."),e
        assert 1 == 0
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

    return saver



if __name__ == '__main__':
    image = tf.placeholder(tf.float32, [None,None,None,3])
    with tf.variable_scope("vgg_16"):
        vgg = Vgg16()
        vgg.build(image)
    restore(vgg, "vgg16.ckpt")
