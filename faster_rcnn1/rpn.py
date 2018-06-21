
import tensorflow as tf
import numpy as np
from anchor_layer import gen_target
import tensorflow.contrib.slim as slim

class RPN:
    def __init__(self, num_class, is_training, weights_initializer, batch_size):
        self.num_class = num_class
        self.is_training = is_training
        self.batch_size = batch_size
        self.weights_initializer = weights_initializer

    def build(self, feature, gt, im_info, num_anchors, anchor_list):
        print "feature shape:",feature
        rpn = slim.conv2d(feature, 512, [3, 3], trainable=self.is_training, weights_initializer=self.weights_initializer, scope="rpn_conv/3x3")
        print "feature shape after conv:",feature.shape

        with tf.variable_scope('cls'):
            rpn_cls_score = slim.conv2d(rpn, 2*num_anchors, [1, 1], trainable=self.is_training, padding='VALID', weights_initializer=self.weights_initializer, scope="rpn_cls_score")
            print "cls shape:",rpn_cls_score

            rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2,"rpn_cls_score_reshape")
            rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
            rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
#for proposal
            rpn_cls_prob = self._reshape_layer(rpn_cls_score, 2*num_anchors,"rpn_cls_prob")

        with tf.variable_scope('bbox'):
            rpn_bbox_pred = slim.conv2d(feature, 4*num_anchors, [1, 1], trainable=self.is_training, padding='VALID', activation_fn=None,
                    weights_initializer=self.weights_initializer, scope="rpn_bbox_pred")
#for proposal
            print "bbox shape:",rpn_bbox_pred

        labels, bbox_targets,inside_weights,outside_weights = gen_target(feature, gt, im_info, num_anchors, anchor_list, self.batch_size)

        self.rpn_cls_score = rpn_cls_score
        self.rpn_cls_score_reshape = rpn_cls_score_reshape
        self.rpn_cls_pred = rpn_cls_pred
        self.rpn_cls_prob = rpn_cls_prob
        self.rpn_bbox_pred = rpn_bbox_pred


        self.cls_logit = rpn_cls_score_reshape
        self.cls_label = labels

        self.bbox_logit = rpn_bbox_pred
        self.bbox_target = bbox_targets
        self.bbox_target_in_weight = inside_weights
        self.bbox_target_out_weight = outside_weights

        return rpn_cls_prob, rpn_bbox_pred

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        print ("====== reshape:",bottom)
        with tf.variable_scope(name) as scope:
# change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
# then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
             tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
# then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)


    def get_loss(self):
        return self.rpn_loss_cls, self.rpn_loss_box

    def conv(self,x, shape, stride, initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
        with tf.variable_scope('conv'):
            weights = tf.get_variable('conv_weights', shape=shape, dtype='float', initializer=initializer)
            print weights
            x = tf.nn.conv2d(x, weights, [1,stride,stride,1], padding='SAME')
            return x

    def bn(self,x, mode='train'):
       return tf.contrib.layers.batch_norm(inputs=x,
               decay=0.95,
               center=True,
               scale=True,
               is_training=(mode=='train'),
               updates_collections=None,
               scope=('batch_norm'))


#train()
