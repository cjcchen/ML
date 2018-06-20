import numpy as np
import math
import tensorflow as tf
import random

from utils.cython_bbox import bbox_overlaps
from model.config import cfg

RPN_POSITIVE_WEIGHT = -1.0
RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
RPN_NEGATIVE_OVERLAP = 0.3
RPN_POSITIVE_OVERLAP = 0.7
RPN_FG_FACTOR = 0.5
RPN_BBOX_LAMBDA = 10.0

random.seed(1234)
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def gen_target(image, gt_boxes, image_raw_size, num_anchors, anchor_list, batch_size):
    #print image.shape
    #print gt_boxes.shape
    #print image_raw_size.shape
    #print anchor_size
    #print base_anchors
    labels, bbox_targets,inside_weights,outside_weights = \
            tf.py_func(gen_target_py, [image, gt_boxes, image_raw_size, anchor_list, batch_size],
                    [tf.float32, tf.float32, tf.float32, tf.float32])

    labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'rpn_labels')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'rpn_bbox_targets')
    inside_weights = tf.convert_to_tensor(inside_weights , name = 'rpn_bbox_inside_weights')
    outside_weights = tf.convert_to_tensor(outside_weights , name = 'rpn_bbox_outside_weights')

    labels.set_shape([1, 1, None, None])
    bbox_targets.set_shape([1, None, None, num_anchors * 4])
    inside_weights.set_shape([1, None, None, num_anchors * 4])
    outside_weights.set_shape([1, None, None, num_anchors * 4])

    return labels, bbox_targets,inside_weights,outside_weights

def gen_target_py(image, gt_box, image_raw_size, anchor_list = None, batch_size = 128):
    '''
    img_scale = image_size/features.size
    original image[800,600], 800 is height, 600 is weight
    after zfnet[51,39]
    image_scale = 16

    gt_box is the target set
    '''
    image = image[0]
    gt_box = gt_box
    image_raw_size = image_raw_size
    #print "box shape:",gt_box.shape

    #anchor_list=get_anchors(image, image_raw_size, anchor_size, base_anchors)
    #print ("anchor list:",anchor_list)

    label, bbox_label, in_weight, out_weight = get_label(anchor_list, gt_box, image_raw_size, batch_size)
    #print label.shape, image.shape
    label=label.reshape([1,image.shape[0],image.shape[1],9]).astype(np.float32)
    bbox_target=bbox_label.reshape([1,image.shape[0], image.shape[1], -1]).astype(np.float32)
    in_weight=in_weight.reshape([1,image.shape[0], image.shape[1], -1]).astype(np.float32)
    out_weight=out_weight.reshape([1,image.shape[0], image.shape[1], -1]).astype(np.float32)
    return label, bbox_target, in_weight, out_weight


def get_label(anchor_list,gt_box, image_raw_size, batch_size):
#class label
    over_lap_matrix = np.zeros([len(anchor_list), len(gt_box)])
    label = np.zeros(len(anchor_list))
    label.fill(-1)
    '''
    for j,box in enumerate(gt_box):
        max_area = 0.0
        index = -1
        for i, anchor in enumerate(anchor_list):
            if in_image(anchor, image_raw_size):
                area = over_lap(anchor,box)
                if label[i] != 2:
                    if area > RPN_POSITIVE_OVERLAP:
                        label[i] = 1
                    elif area < RPN_NEGATIVE_OVERLAP:
                        label[i] = 0

                if max_area < area:
                    max_area = area
                    index = i
                over_lap_matrix[i,j] = area
        if index >-1:
            label[over_lap_matrix[:,j] == max_area] = 2
            #for i, anchor in enumerate(anchor_list):
            #    if over_lap_matrix[i,j] == max_area:
            #        label[i] = 2
    label[label==2]=1
    '''
    for i, anchor in enumerate(anchor_list):
        max_area = -1.0
        index = -1
        for j,box in enumerate(gt_box):
            if in_image(anchor, image_raw_size):
                area = over_lap(anchor,box)
                if max_area < area:
                    max_area = area
                    index = i
                over_lap_matrix[i,j] = area
            #if i > 1400 and i < 1410:
            #    print ("my over lap:",i,j,anchor, box, over_lap_matrix[i,j], max_area, index)
        if index >-1:
            if max_area >= RPN_POSITIVE_OVERLAP:
                label[i] = 1
            elif max_area < RPN_NEGATIVE_OVERLAP:
                label[i] = 0
            #print ("!!inside index:",i,label[i], max_area)
            #for i, anchor in enumerate(anchor_list):
            #    if over_lap_matrix[i,j] == max_area:
            #        label[i] = 2

    '''
    for i, anchor in enumerate(anchor_list):
        max_area = 0.0
        index = -1
        for j,box in enumerate(gt_box):
                area = over_lap_matrix[i,j]
                if max_area < area:
                    max_area = area
                    index = i
                over_lap_matrix[i,j] = area
        if index >-1:
            if max_area >= RPN_POSITIVE_OVERLAP:
                label[i] = 1
            elif max_area < RPN_NEGATIVE_OVERLAP:
                label[i] = 0
            #for i, anchor in enumerate(anchor_list):
            #    if over_lap_matrix[i,j] == max_area:
            #        label[i] = 2
    '''

    for j,box in enumerate(gt_box):
        max_area = 0.0
        index = -1
        for i, anchor in enumerate(anchor_list):
            area = over_lap_matrix[i,j]
            if max_area < area:
                max_area = area
                index = i
        if index >-1:
            for i in range(len(anchor_list)):
                if over_lap_matrix[i,j] == max_area:
                    label[i] = 1
            #        print ("!!~~inside index:",i, label[i], max_area)
            #label[over_lap_matrix[:,j] == max_area] = 1
    #print ("1548:",label[1548])

    '''
    overlaps = over_lap_matrix
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(anchor_list)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        label[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

      # fg label: for each gt, anchor with highest overlap
    label[gt_argmax_overlaps] = 1

      # fg label: above threshold IOU
    label[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        label[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    '''

    fg_num = int(RPN_FG_FACTOR * batch_size)
    fg_index = np.where(label == 1)[0]
    #print ("fg num:",fg_num, len(fg_index))
    if len(fg_index) > fg_num:
        remove_index = np.random.choice(fg_index, size=(len(fg_index) - fg_num), replace=False)
        label[remove_index] = -1
    #print ("1548:",label[1548])

    bg_num = batch_size - np.sum(label==1)
    bg_index = np.where(label == 0)[0]
    #print("bg num:",len(bg_index), bg_num)
    if len(bg_index) > bg_num:
        remove_index = np.random.choice(bg_index, size=(len(bg_index) - bg_num), replace=False)
        label[remove_index] = -1
        #print ("get:",remove_index, len(remove_index))
    #print ("1548:",label[1548])

    in_weight = np.zeros([len(anchor_list),4], dtype=np.float32)
    out_weight = np.zeros([len(anchor_list),4], dtype=np.float32)

#bbox label
    dx = np.zeros(len(anchor_list))
    dy = np.zeros(len(anchor_list))
    dw = np.zeros(len(anchor_list))
    dh = np.zeros(len(anchor_list))

    num_examples = np.sum(label>=0)
    for i, anchor in enumerate(anchor_list):
        w = anchor[2]-anchor[0]+1.0
        h = anchor[3]-anchor[1]+1.0
        center_x = anchor[0] + w/2
        center_y = anchor[1] + h/2

        max_gt = over_lap_matrix[i].argmax()
        if in_image(anchor, image_raw_size):
            target = gt_box[max_gt]

            target_w = target[2]-target[0]+1.0
            target_h = target[3]-target[1]+1.0
            target_center_x = target[0] + target_w/2
            target_center_y = target[1] + target_h/2

            dx[i] = (target_center_x-center_x)/w
            dy[i] = (target_center_y-center_y)/h
            dw[i] = np.log(target_w/w)
            dh[i] = np.log(target_h/h)
            if label[i]==1:
                in_weight[i] = [1.0]*4
                out_weight[i] = [1.0/num_examples]*4
            if label[i]==0:
                out_weight[i] = [1.0/num_examples]*4
    bbox_target = np.vstack( (dx,dy,dw,dh) ).transpose()
    #print ("1548:",label[1548])
    return label, bbox_target,in_weight, out_weight

def in_image(anchor, image_raw_size):
    x0 = anchor[0]
    y0 = anchor[1]
    x1 = anchor[2]
    y1 = anchor[3]
    #print "in image:",x0, y0,x1,y1, image_raw_size[0], image_raw_size[1]
    return x0>=0 and y0 >=0 and x1 < image_raw_size[1] and y1 < image_raw_size[0]

def over_lap(anchor, target):
    x0 = anchor[0]
    y0 = anchor[1]
    x1 = anchor[2]
    y1 = anchor[3]

    x2 = target[0]
    y2 = target[1]
    x3 = target[2]
    y3 = target[3]

    if x0 > x3 or x1 < x2 or y0 > y3 or y1 < y2:
        return 0
    #print "over lap:",anchor, target
    over_lap_area = (min(x3,x1)-max(x2,x0)+1)*(min(y3,y1)-max(y2,y0)+1)
    total_area = (y1-y0+1)*(x1-x0+1)+(y3-y2+1)*(x3-x2+1)-over_lap_area
    return float(over_lap_area)/total_area

def smooth(x, sigma=3.0):
    '''
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
    (https://arxiv.org/pdf/1504.08083v2.pdf)

                    0.5 * (sigma * x)^2         if |x| < 1/sigma^2
    smoothL1(x) = {
                    |x| - 0.5/sigma^2           otherwise
    '''
    conditional = tf.less(tf.abs(x), 1/sigma**2)
    close = 0.5 * (sigma * x)**2
    far = tf.abs(x) - 0.5/sigma**2
    return tf.where(conditional, close, far)

def get_bbox_loss(predict, target, in_weight, out_weight):
    diff = predict - target
    diff = tf.multiply(diff,in_weight)
    r = smooth(diff)
    loss = tf.reduce_sum(tf.multiply(r,out_weight))
    return loss * RPN_BBOX_LAMBDA

def get_cls_loss(predict, target):
    '''
    [1,wieght, height, 9*2]
    '''

    rpn_cls_score = tf.reshape(predict, [-1, 2])
    rpn_label = tf.reshape(target, [-1])

    rpn_select = tf.where(tf.not_equal(rpn_label, -1))

    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

from utils.blob import prep_im_for_blob, im_list_to_blob
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
def _get_image_blob(file_name):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  processed_ims = []
  im = cv2.imread(file_name)
  target_size = 600
  max_size = 1000
  im, im_scale = prep_im_for_blob(im, PIXEL_MEANS, target_size, max_size)
  processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  return blob

if __name__ =='__main__':
    #images=np.zeros((1,51,39,18))
    #bbox=np.zeros((51,39,36))
    #gt_boxes = np.array([(10,10,20,20,3),(45,45,60,75,4)])

    im = _get_image_blob("./VOCdevkit2007/VOC2007/JPEGImages/000005.jpg")
    im_info = np.array([ 600,800,1.60000002])
    #gt_box = np.array([[ 419.20001221  336.          516.79998779  540.79998779    9.        ]
    #         [ 262.3999939   420.79998779  403.20001221  593.59997559    9.        ]
    #          [ 384.          308.79998779  470.3999939   476.79998779    9.        ]])

    ik = tf.placeholder(tf.float32, [1,None,None,18])
    gt = tf.placeholder(tf.float32, [None,5])
    #bbx = tf.placeholder(tf.float32, [1, None,None,36])
    img_size = tf.placeholder(tf.float32, [3])

    labels, bbox_target,inside_weights,outside_weights = gen_target(ik, gt, img_size)

    #print "label:",labels.shape
    #print "bbox:",bbox_target.shape
    #cls_loss = get_cls_loss(ik, labels)
    bbox_loss = get_bbox_loss(bbx, bbox_target, inside_weights, outside_weights)
    cls_loss = get_cls_loss(ik,labels)
    print "loss:",bbox_loss
    with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            feed_dict_train = {ik:images, gt:gt_boxes, bbx:bbox, img_size:im_info}
            [d]=sess.run([cls_loss], feed_dict=feed_dict_train)
            print "d:",d*100
            print d.shape
            print "0:",np.where(d==0)
            print "1:",np.where(d!=0)
            print d[d!=0]*1000

'''
images=np.zeros((1,51,39,18))
feature=np.zeros([51,39,9*4])
gt_boxes = np.array([(10,10,20,20,3),(45,45,60,75,4)])
im_info = np.array([ 800.,  600.,    3.], dtype=np.int32)

label, bbox_target, in_w, out_w=gen_target(images, gt_boxes, im_info)
print "output:"
print label.shape
print bbox_target.shape
print in_w.shape
print out_w.shape
#print bbox_target[bbox_target>0,:]
#print np.where(label>0)

print get_loss(feature, bbox_target, in_w, out_w)

'''
