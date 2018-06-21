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
        if is_training:
            reuse = None
        else:
            reuse = True
        with tf.variable_scope(self._scope, reuse = reuse):
            return self._image_to_head(rgb,is_training)

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
