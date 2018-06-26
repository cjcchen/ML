from rpn import RPN
import numpy as np
from proposal import Proposal
from util import get_anchors
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

POOLING_SIZE=7
RPN_BBOX_LAMBDA = 10.0
WEIGHT_DECAY = 0.0001
BIAS_DECAY = False
DOUBLE_BIAS = True
MOMENTUM = 0.9

class FasterRCNN:
    def __init__(self, cnn_net, num_class, batch_size=1, is_training=True):
        self._scope = 'vgg_16'
        if not is_training:
            self.reuse = tf.AUTO_REUSE
        else:
            self.reuse = None

        with tf.variable_scope(self._scope, self._scope, reuse=self.reuse):
            self.image = tf.placeholder(tf.float32, [1,None,None,3])
            self.gt_boxes = tf.placeholder(tf.float32, [None,5])
            self.im_info = tf.placeholder(tf.float32, [3])

        self.cnn_net = cnn_net
        self.batch_size=batch_size
        self.num_class = num_class
        self.is_training = is_training

        self._feat_stride=16

        self.anchor_ratio = [0.5,1,2]
        self.base_anchors = [8,16,32]
        self.num_anchors = len(self.anchor_ratio) * len(self.base_anchors)

        if is_training:
            self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            self.initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            self.initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        self.rpn = RPN(self.num_class, is_training, self.initializer, self.batch_size)
        self.proposal = Proposal(self.num_class, is_training,  self.initializer, self.batch_size)

    def build(self, mode):
        weights_regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        biases_regularizer = tf.no_regularizer

        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer,
                    biases_initializer=tf.constant_initializer(0.0)):

            self.cnn_net.build(self.image, is_training = self.is_training, mode=mode)

            self.feature_input = self.cnn_net.get_output()

            with tf.variable_scope(self._scope, self._scope, reuse=self.reuse):
                rois = self.build_proposal()
                pool5 = self._crop_pool_layer(self.feature_input, rois, "crop")

                self.build_tail(pool5)

            if mode=='train' or mode == 'val':
                self.build_loss()
                self.lr, self.train_op = self.build_train_op()


    def build_proposal(self):
        self.anchor_list = get_anchors(self.feature_input, self.im_info, self.anchor_ratio, self.base_anchors)
        self.rpn_layer()
        return self.proposal_layer()

    def rpn_layer(self):
        self.rpn_cls, self.rpn_bbox = self.rpn.build(self.feature_input, self.gt_boxes, self.im_info, self.num_anchors, self.anchor_list)

    def proposal_layer(self):
        self.proposal.build(self.rpn_cls, self.rpn_bbox, self.gt_boxes, self.im_info, self.num_anchors, self.anchor_list)
        return self.proposal.rois

    def build_tail(self, rois):
        flatten_rois = slim.flatten(rois, scope='flatten')
        fc5 = slim.fully_connected(flatten_rois, 4096, scope="fc6")
        if self.is_training:
            fc5 = slim.dropout(fc5, keep_prob=0.5, is_training=True, scope='dropout6')

        fc6 = slim.fully_connected(fc5, 4096, scope="fc7")
        if self.is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout7')

        self.cls_logit = slim.fully_connected(fc6, self.num_class,
                                        weights_initializer=self.initializer,
                                       trainable=self.is_training,
                                       activation_fn=None, scope='cls_logit')
        self.cls_prob = tf.nn.softmax(self.cls_logit)
        self.cls_pred = tf.argmax(self.cls_prob, axis=1, name="cls_pred")

        self.bbox_logit = slim.fully_connected(fc6, self.num_class*4,
                                     weights_initializer=self.initializer_bbox,
                                     trainable=self.is_training,
                                     activation_fn=None, scope='bbox_logit')
        self.bbox_delta_pred = self.bbox_logit


    def _crop_pool_layer(self, bottom, rois, name):
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
# Get the normalized coordinates of bounding boxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride)
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride)
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
# Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')


    def build_loss(self):
        rpn_cls_loss = self.get_cls_loss(self.rpn.cls_logit, self.rpn.cls_label)
        self.rpn_cross_entropy = rpn_cls_loss

        rpn_bbox_loss = self._smooth_l1_loss(self.rpn.bbox_logit, self.rpn.bbox_target, self.rpn.bbox_target_in_weight, self.rpn.bbox_target_out_weight, sigma=3.0, dim=[1,2,3])

        self.rpn_loss_box = rpn_bbox_loss

# RCNN, class loss
        cls_label = tf.to_int32(self.proposal.cls_label, name="to_int32")
        cls_label = tf.reshape(cls_label, [-1])
        print ("loss:",cls_label.shape)
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_logit, labels=cls_label))#[-1, 21], [-1,1]
        self.cross_entropy = cls_loss

        bbox_loss = self._smooth_l1_loss(self.bbox_logit, self.proposal.bbox_target, self.proposal.bbox_target_in_weight,self.proposal.bbox_target_out_weight)

        self.loss_box = bbox_loss

        loss = rpn_cls_loss + rpn_bbox_loss + cls_loss + bbox_loss
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        self.loss = loss + regularization_loss
        #return self.loss

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
          out_loss_box,
          axis=dim
        ))
        return loss_box

    def build_train_op(self):
        lr = tf.Variable(0.01, trainable=False)
        i_global_op=tf.train.get_or_create_global_step()
        self.global_op = i_global_op
        self.optimizer = tf.train.MomentumOptimizer(lr, MOMENTUM)

       # Compute the gradients with regard to the loss
        gvs = self.optimizer.compute_gradients(self.loss)
        # Double the gradient of the bias if set
        if DOUBLE_BIAS:
           final_gvs = []
           with tf.variable_scope('Gradient_Mult') as scope:
            for grad, var in gvs:
                scale = 1.
                if '/biases:' in var.name:
                    scale *= 2.
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gvs.append((grad, var))
           train_op = self.optimizer.apply_gradients(final_gvs, global_step=i_global_op)

           for grad, var in final_gvs:
            if grad is not None :
                tf.summary.histogram(var.op.name + '/gradients', grad)
        else:
           train_op = self.optimizer.apply_gradients(gvs, global_step=i_global_op)
           for grad, var in gvs:
            if grad is not None :
                tf.summary.histogram(var.op.name + '/gradients', grad)

        self.summary_op = tf.summary.merge_all()
        return lr, train_op

    def get_cls_loss(self, predict, target):
        '''
        [1,wieght, height, 9*2]
        '''
        rpn_cls_score = tf.reshape(predict, [-1, 2])
        rpn_label = tf.reshape(target, [-1])

        rpn_select = tf.where(tf.not_equal(rpn_label, -1))

        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    def train_step(self, sess, image, gt_boxes, im_info):
        loss,lr,global_step, _, summary_str = sess.run( [self.loss, self.lr, self.global_op, self.train_op, self.summary_op],
                feed_dict={self.image:image, self.gt_boxes:gt_boxes, self.im_info:im_info.reshape(-1)} )
        import math
        assert not math.isnan(loss)

        return loss, lr, global_step, summary_str

    def get_loss(self, sess, image, gt_boxes, im_info):
        loss = sess.run( self.loss, feed_dict={self.image:image, self.gt_boxes:gt_boxes, self.im_info:im_info.reshape(-1)} )
        import math
        assert not math.isnan(loss)

        return loss

    def predict(self,sess,image, im_info):
#score and delta bbox
        score, delta_bbox, rois = sess.run( [self.cls_prob, self.bbox_delta_pred, self.proposal.rois],
                feed_dict={self.image:image, self.im_info:im_info.reshape(-1)} )

        bbox = self.proposal.bbox_target_inv(rois, delta_bbox, im_info)

        assert score.shape[1] == self.num_class
        image_score_list = []
        image_bbox_list = []
        thresh = 0
        for i in range(self.num_class):
            inds = np.where( score[:,i] > thresh )[0]
            image_score = score[inds,i]
            image_bbox = bbox[inds, i*4:(i+1)*4]
            image_score_list.append(image_score)
            image_bbox_list.append(image_bbox)

        image_scores = np.hstack([image_score_list[i] for i in range(1, self.num_class)])
        image_thresh = np.sort(image_scores)[-3]
        print ("thresh:",image_thresh)
        res_score = []
        res_bbox = []
        for i in range(1, self.num_class):
            #print ("class i:",i,image_score_list[i])
            keep = image_score_list[i]>= image_thresh

            image_score = image_score_list[i][keep]
            image_bbox = image_bbox_list[i][keep]
            res_score.append(image_score)
            res_bbox.append(image_bbox)
            #print ("get sore:",keep, i,image_score_list[i][keep].shape)
        return res_score, res_bbox


    def assign_lr(self, sess, rate):
        sess.run(tf.assign(self.lr, rate))

if __name__ == '__main__':
    image = tf.placeholder(tf.float32,[None,None,None,3])

    from vgg16 import Vgg16
    cnn_net = Vgg16()
    faster_rcnn = FasterRCNN(cnn_net)
    faster_rcnn.build(21)


