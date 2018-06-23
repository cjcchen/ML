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

def gen_target(image, gt_boxes, image_raw_size, num_anchors, anchor_list, batch_size):
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

    label, bbox_label, in_weight, out_weight = get_label(anchor_list, gt_box, image_raw_size, batch_size)
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

    inside_idx = np.where(
            (anchor_list[:,0]>=0 )&
            (anchor_list[:,1] >=0 )&
            (anchor_list[:,2] < image_raw_size[1] )&
            (anchor_list[:,3] < image_raw_size[0])
            )[0]

    over_lap_matrix = bbox_overlaps(
        np.ascontiguousarray(anchor_list, dtype=np.float),
        np.ascontiguousarray(gt_box, dtype=np.float))

    anchor_max_idx = over_lap_matrix.argmax(axis=1)
    over_lap_max = over_lap_matrix[np.arange(len(anchor_list)), anchor_max_idx ]

    label[over_lap_max >= RPN_POSITIVE_OVERLAP] = 1
    label[(over_lap_max < RPN_NEGATIVE_OVERLAP)] = 0
    for i in range(len(anchor_list)):
        if i not in inside_idx:
            label[i] = -1

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
        if index >-1:
            if max_area >= RPN_POSITIVE_OVERLAP:
                label[i] = 1
            elif max_area < RPN_NEGATIVE_OVERLAP:
                label[i] = 0
            inside_index.append(i)
    '''


    gt_max_index = over_lap_matrix.argmax(axis=0)
    gt_max = over_lap_matrix[ gt_max_index, np.arange(over_lap_matrix.shape[1])]
    gt_max_index = np.where(over_lap_matrix==gt_max)[0]
    label[ gt_max_index ] = 1
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
    '''

    fg_num = int(RPN_FG_FACTOR * batch_size)
    fg_index = np.where(label == 1)[0]
    if len(fg_index) > fg_num:
        remove_index = np.random.choice(fg_index, size=(len(fg_index) - fg_num), replace=False)
        label[remove_index] = -1

    bg_num = batch_size - np.sum(label==1)
    bg_index = np.where(label == 0)[0]
    if len(bg_index) > bg_num:
        remove_index = np.random.choice(bg_index, size=(len(bg_index) - bg_num), replace=False)
        label[remove_index] = -1

    in_weight = np.zeros([len(anchor_list),4], dtype=np.float32)
    out_weight = np.zeros([len(anchor_list),4], dtype=np.float32)

#bbox label
    dx = np.zeros(len(anchor_list))
    dy = np.zeros(len(anchor_list))
    dw = np.zeros(len(anchor_list))
    dh = np.zeros(len(anchor_list))


    ws = anchor_list[inside_idx,2]-anchor_list[inside_idx,0]+1.0
    hs = anchor_list[inside_idx,3]-anchor_list[inside_idx,1]+1.0
    center_xs = anchor_list[inside_idx,0] + ws/2
    center_ys = anchor_list[inside_idx,1] + hs/2

    gt_target = gt_box[ anchor_max_idx ]

    target_w = gt_target[inside_idx,2]-gt_target[inside_idx,0]+1.0
    target_h = gt_target[inside_idx,3]-gt_target[inside_idx,1]+1.0
    target_center_x = gt_target[inside_idx,0] + target_w/2.0
    target_center_y = gt_target[inside_idx,1] + target_h/2.0

    dx[inside_idx] = (target_center_x-center_xs)/ws
    dy[inside_idx] = (target_center_y-center_ys)/hs
    dw[inside_idx] = np.log(target_w/ws)
    dh[inside_idx] = np.log(target_h/hs)

    num_examples = np.sum(label>=0)

    in_weight[label==1] = [1.0]*4
    out_weight[label==1] = [1.0/num_examples]*4
    out_weight[label==0] = [1.0/num_examples]*4

    '''
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
    '''
    bbox_target = np.vstack( (dx,dy,dw,dh) ).transpose()
    return label, bbox_target,in_weight, out_weight

def in_image(anchor, image_raw_size):
    x0 = anchor[0]
    y0 = anchor[1]
    x1 = anchor[2]
    y1 = anchor[3]
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

def _get_image_blob(file_name):
  from utils.blob import prep_im_for_blob, im_list_to_blob
  PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
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

