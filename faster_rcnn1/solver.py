
from data_input import combined_roidb
from vgg16 import Vgg16
from faster_rcnn import FasterRCNN
import numpy as np
import tensorflow as tf
import random
from roi_data_layer.layer import RoIDataLayer

LEARNING_RATE = 0.001
GAMMA = 0.1

class Solver():
    def __init__(self, imdb, roidb, pretrain_model):
        self.imdb = imdb
        self.roidb = roidb

        self.pretrain_model = pretrain_model
        self.cnn_net = Vgg16()
        self.faster_rcnn = FasterRCNN(self.cnn_net, self.imdb.num_classes, batch_size=256, is_training=True)
        self.faster_rcnn.build()

    def train_net(self,max_iters=70000):
        """Train a Faster R-CNN network."""
        roidb = self.filter_roidb(self.roidb)

        load_vars = []
        train_vars = tf.global_variables()
        for var in train_vars:
            if var.name.startswith("vgg_16"):
                load_vars.append(var)


        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        saver = tf.train.Saver(load_vars)

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


        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(pretrained_model)
        print ("var keep dic:",var_keep_dic)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.cnn_net.get_variables_to_restore(variables, var_keep_dic)
        #variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
        print ("var to restore:",variables_to_restore)
        #print ("vgg variables:",variables, self.pretrained_model)
        restorer = tf.train.Saver(variables_to_restore)

        #load_vars = []
        #for var in var_keep_dic:
        #    load_vars.append(var)
        #print (load_vars)

        #restorer = tf.train.Saver(load_vars)
        restorer.restore(sess, pretrained_model)
        #g_vars= tf.global_variables()

        print('Loaded.')
      # Need to fix the variables before loading, so that the RGB weights are changed to BGR
      # For VGG16 it also changes the convolutional weights fc6 and fc7 to
      # fully connected weights
        self.cnn_net.fix_variables(sess, pretrained_model)
      #self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')
        #last_snapshot_iter = 0
        #rate = cfg.TRAIN.LEARNING_RATE
        #stepsizes = list(cfg.TRAIN.STEPSIZE)

        return

    def train_model(self, sess, max_iters):
        print "train:", self.roidb
        # Build data layers for both training and validation set
        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        iter = 0
        rate = LEARNING_RATE
        next_step = [0, 20]
        # Make sure the lists are not empty
        while iter < max_iters + 1:
            if len(next_step) > 0 && iter == next_step[0]:
                self.faster_rcnn.assign_lr(sess, rate)
                next_step=next_step[1:]
                print ("next step:",next_step)
                rate *= GAMMA

            blobs = self.data_layer.forward()
            image = blobs['data']
            gt_boxes = blobs['gt_boxes']
            im_info = blobs['im_info']

            self.faster_rcnn.train_step( sess, image, gt_boxes, im_info)
            iter+=1

    def filter_roidb(self, roidb):
      """Remove roidb entries that have no usable RoIs."""
      FG_THRESH = 0.5
      BG_THRESH_HI = 0.5
      BG_THRESH_LO = 0.0
      def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < BG_THRESH_HI) &
                           (overlaps >= BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

      num = len(roidb)
      filtered_roidb = [entry for entry in roidb if is_valid(entry)]
      num_after = len(filtered_roidb)
      print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                         num, num_after))
      return filtered_roidb




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
