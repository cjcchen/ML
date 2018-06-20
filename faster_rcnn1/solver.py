
from data_input import combined_roidb
from data_input import filter_roidb
from vgg16 import Vgg16
from faster_rcnn import FasterRCNN
import numpy as np
import tensorflow as tf
import random
import os
from roi_data_layer.layer import RoIDataLayer

LEARNING_RATE = 0.001
GAMMA = 0.1
SAVE_STEP = 20

class Solver():
    def __init__(self, imdb, roidb, pretrain_model):
        self.imdb = imdb
        self.roidb = roidb
        self.pretrain_model = pretrain_model
        self.model_dir = "./model_log/"
        self.cnn_net = Vgg16()
        self.faster_rcnn = FasterRCNN(self.cnn_net, self.imdb.num_classes, batch_size=256, is_training=True)

        self.faster_rcnn.build()

    def train_net(self,max_iters=70000):
        """Train a Faster R-CNN network."""
        roidb = filter_roidb(self.roidb)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        with tf.Session(config=tfconfig) as sess:
            self.initialize(sess, self.pretrain_model)
            self.train_model(sess, max_iters)

    def get_variables_in_checkpoint_file(self, file_name):
        from tensorflow.python import pywrap_tensorflow
        try:
          reader = pywrap_tensorflow.NewCheckpointReader(file_name)
          var_to_shape_map = reader.get_variable_to_shape_map()
          return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
          print(str(e))
          if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

    def initialize(self, sess, pretrained_model):
        tf.set_random_seed(1234)
        random.seed(1234)

        self.saver = tf.train.Saver(max_to_keep=100000)
        try:
            checkpoint_dir = self.model_dir
            print("Trying to restore last checkpoint ...:",checkpoint_dir)
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            self.saver.restore(sess, save_path=last_chk_path)
            print("restore last checkpoint %s done"%checkpoint_dir)
        except Exception as e:
            print("Failed to restore checkpoint. Initializing variables instead."),e

            # Initial file lists are empty
            # Fresh train directly from ImageNet weights
            print('Loading initial model weights from {:s}'.format(pretrained_model))
            variables = tf.global_variables()
            # Initialize all variables first
            sess.run(tf.variables_initializer(variables, name='init'))
            var_keep_dic = self.get_variables_in_checkpoint_file(pretrained_model)
            variables_to_restore = self.cnn_net.get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, pretrained_model)

            print('Loaded.')
            self.cnn_net.fix_variables(sess, pretrained_model)

            print('Fixed.')
            return

    def save_model(self, sess, global_step):
        self.saver.save(sess, os.path.join(self.model_dir,'cp'), global_step=global_step)
        print ("save model:",os.path.join(self.model_dir,'cp'))

    def train_model(self, sess, max_iters):
        print "train:", self.roidb
        # Build data layers for both training and validation set
        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        iter = 0
        rate = LEARNING_RATE
        next_step = [0, 20]
        # Make sure the lists are not empty
        while iter < max_iters + 1:
            if len(next_step) > 0 and iter == next_step[0]:
                self.faster_rcnn.assign_lr(sess, rate)
                next_step=next_step[1:]
                print ("next step:",next_step)
                rate *= GAMMA

            blobs = self.data_layer.forward()
            image = blobs['data']
            gt_boxes = blobs['gt_boxes']
            im_info = blobs['im_info']

            loss, lr, global_step = self.faster_rcnn.train_step( sess, image, gt_boxes, im_info)
            iter+=1
            print ("===== loss:",loss, "lr:",lr, "global step:",global_step)

            if iter % SAVE_STEP == 0:
                self.save_model(sess, global_step)

def train():

    imdb, roidb = combined_roidb('voc_2007_trainval')

    from roi_data_layer.minibatch import get_minibatch
    for roi in roidb:
        if not roi['flipped']:
            print roi
            print roi['gt_classes']
            print roi['max_classes']
            print roi['boxes']


    sv = Solver(imdb, roidb, "tf-faster-rcnn/data/imagenet_weights/vgg16.ckpt")
    sv.train_net(30)


train()
