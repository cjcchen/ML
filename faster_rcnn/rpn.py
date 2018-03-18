from component import component
import anchor 
import tensorflow as tf
from component import network_base

class RPN(network_base.BaseNet):
    def __init__(self, pre_model, images, gt_box, img_size, pre_fix):
        network_base.BaseNet.__init__(self,pre_fix)
        
        self.pre_model = pre_model
        self.images = images
        self.gt_box = gt_box
        self.img_size = img_size

        self.build_model()
    

    def build_model(self):
        with tf.variable_scope('rpn'):
            conv_x = component.conv(self.images,ksize=3,output_channel=256,stride=1)

        with tf.variable_scope('cls'):
            self.cls_score = component.conv(conv_x, ksize=1, output_channel=9*2, stride=1)

        with tf.variable_scope('bbox'):
            self.bbox_pred = component.conv(conv_x, ksize=1, output_channel=9*4, stride=1)

        with tf.variable_scope('cls_target'):
            labels, bbox_target,inside_weights,outside_weights = \
                anchor.rpn_anchor_layer(self.cls_score, self.gt_box, self.img_size, 
                        feat_stride = 16, anchor_scales = [128, 256, 512])

        with tf.variable_scope('loss'):
            cls_loss = anchor.get_cls_loss(self.cls_score, labels)
            bbox_loss = anchor.get_bbox_loss(self.bbox_pred, bbox_target, inside_weights, outside_weights)

            self.loss = bbox_loss + cls_loss
            print "self.loss:",self.loss


