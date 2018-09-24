
from vgg16 import Vgg16
from faster_rcnn import FasterRCNN
import numpy as np
import tensorflow as tf
import random
import time
import os
from data_layer.minibatch import get_minibatch
from data_layer.data_input import combined_roidb

LEARNING_RATE = 0.001
GAMMA = 0.1
SAVE_STEP = 200
BATCH_SIZE = 256

tf.flags.DEFINE_string("model_path", "/home/tusimple/junechen//ml_data/model/faster_rcnn/model/",
                    "Where the training/test data is stored.")
tf.flags.DEFINE_string("log_path", "/home/tusimple/junechen//ml_data/model/faster_rcnn/log/",
                    "Model output directory.")
tf.flags.DEFINE_string("val_log_path", "/home/tusimple/junechen/ml_data/model/faster_rcnn/val_log/",
                    "Model output directory.")
FLAGS = tf.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class Predict():
    def __init__(self, num_classes):
        self.model_dir = FLAGS.model_path
        self.cnn_net = Vgg16()

        with tf.device("/gpu:0"):
            self.faster_rcnn = FasterRCNN(self.cnn_net, num_classes, batch_size=BATCH_SIZE, is_training=False)
            self.faster_rcnn.build(mode='predict')
        self._initialize()


    def predict(self,image, im_info):
        """Train a Faster R-CNN network."""
#allow_soft_placement=True,log_device_placement=True
        return self._predict(image,im_info)

    def _initialize(self):
        tf.set_random_seed(1234)
        random.seed(1234)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tfconfig)

        self.saver = tf.train.Saver(max_to_keep=100000)
        try:
            checkpoint_dir = self.model_dir
            print("Trying to restore last checkpoint ...:",checkpoint_dir)
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            self.saver.restore(self.sess, save_path=last_chk_path)
            print("restore last checkpoint %s done"%checkpoint_dir)
        except Exception as e:
            print e
            raise e

    def _predict(self, image, im_info):
        return self.faster_rcnn.predict(self.sess, image, im_info)

def predict():
    #imdb, roidb = combined_roidb('voc_2007_trainval')
    imdb, roidb = combined_roidb('voc_2007_test', is_training=False)
    print imdb.classes
    #from data_layer.layer import RoIDataLayer
    #data_layer = RoIDataLayer(roidb, imdb.num_classes)
    #for _ in range(10):
    #  blobs = data_layer.forward()
    #  print blobs

    sv = Predict(imdb.num_classes)
    ids=0
    for roi in roidb:
        print "====================================================================="
        print roi
        blobs=get_minibatch([roi], imdb.num_classes)
        image = blobs['data']
        gt_boxes = blobs['gt_boxes']
        im_info = blobs['im_info']
        clss_score, clss_bbox = sv.predict(image, im_info)
        for i,(score, bbox) in enumerate(zip(clss_score, clss_bbox)):
            if len(score)>0:
                print ("class:",i+1, imdb.classes[i+1])
                print "score:",score
                print "bbox:",bbox
        ids+=1
        if ids > 10:
            break

predict()

