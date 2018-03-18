import numpy as np
import tensorflow as tf

RPN_POSITIVE_WEIGHT = -1.0
RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
RPN_NEGATIVE_OVERLAP = 0.3
RPN_POSITIVE_OVERLAP = 0.7
RPN_SIZE = 32
RPN_FG_FACTOR = 0.5
RPN_BBOX_LAMBDA = 10.0

def rpn_anchor_layer(images, gt_boxes, img_size, feat_stride = [16], anchor_scales = [128, 256, 512]):
    '''
    feture img(input), ground_true_box(output), image info, feture img/raw img ratio, 
        anchor size, 16*8, 16*16,16*32(16*16 base)
    images (batch, height, weight, channel)
    ground box(x1,y1,x2,y2,class)
    imgsize (height, weight)
    feat stride ratio for raw img / images
    anchor base anchor size
    '''
    labels, bbox_targets,inside_weights,outside_weights = \
            tf.py_func(anchor_layer_py, [images, gt_boxes, img_size, feat_stride, anchor_scales], 
                    [tf.float32, tf.float32, tf.float32, tf.float32] )

    labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'rpn_labels')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'rpn_bbox_targets')
    inside_weights = tf.convert_to_tensor(inside_weights , name = 'rpn_bbox_inside_weights')
    outside_weights = tf.convert_to_tensor(outside_weights , name = 'rpn_bbox_outside_weights')

    return labels, bbox_targets,inside_weights,outside_weights


def anchor_layer_py(images, gt_boxes, img_size, feat_stride = [16], anchor_scales = [128, 256, 512]):
    assert len(images) == 1
    images = images[0]
    img_size = img_size[0]
    gt_boxes = gt_boxes[0]

    all_anchors, inside_index = get_all_anchor(images, img_size, feat_stride, anchor_scales)
    inside_anchors = all_anchors[inside_index,:]
    bbox, labels = get_target(inside_anchors, gt_boxes)
    target_weight, inside_weights,outside_weights = get_target_weight(inside_anchors, gt_boxes, bbox, labels)

    height,weight,_ = images.shape
    target_weight = enlarge(target_weight, len(all_anchors), inside_index, default_value=0)
    labels = enlarge(labels, len(all_anchors),inside_index, default_value=-1)
    inside_weights = enlarge(inside_weights, len(all_anchors), inside_index, default_value=0)
    outside_weights = enlarge(outside_weights, len(all_anchors), inside_index, default_value=0)

    target_weight=target_weight.reshape(1,height,weight,-1)
    labels=labels.reshape(1, height,weight, -1)
    inside_weights=inside_weights.reshape(1, height,weight, -1)
    outside_weights=outside_weights.reshape(1, height,weight, -1)
    return labels, target_weight,inside_weights,outside_weights

def get_all_anchor(images, im_raw_size, feat_stride, anchor_scales):
    base_anchors = generate_basic_anchors(anchor_scales=anchor_scales) # get the 9 baseic anchor (1*1, 1*2,2*1)
    height,width = images.shape[0:2]
    all_anchor = []
    for y in xrange(0,height):
        for x in xrange(0,width):
            all_anchor.append(base_anchors + [x*feat_stride,y*feat_stride,x*feat_stride,y*feat_stride])
    all_anchor = np.array(all_anchor)
    all_anchor=all_anchor.reshape(-1,4)
    allowed_border = 0
    #inside
    inside_index = np.where(
                    (all_anchor[:, 0] >= -allowed_border) &
                    (all_anchor[:, 1] >= -allowed_border) &
                    (all_anchor[:, 2] < im_raw_size[1] + allowed_border) &  # width
                    (all_anchor[:, 3] < im_raw_size[0] + allowed_border)    # height
                    )[0]

    return all_anchor,inside_index

def generate_basic_anchors(base_size = 16, ratios=[0.5,1,2], anchor_scales=[128,256,512]):
    '''
    create 1*1,1*2,2*1 for 128**2, 256**2, 512**2 by default
    use base * base to create a fix central point
    '''
    b_w = base_size
    b_h = base_size
    b_x = (b_w-1)/2.0
    b_y = (b_h-1)/2.0
    s = b_w*b_h 

    anchor_list=[]
    for r in ratios:
        r_w = np.round(np.sqrt(s/r))
        r_h = np.round(r_w*r) # 1*1, 1*2, 2*1 -> anchor's width and height
        for scale in anchor_scales:
            #enlarge the anchor
            w = r_w * scale/base_size
            h = r_h * scale/base_size
            anchor=np.hstack((b_x-(w-1)/2,b_y-(h-1)/2,b_x+(w-1)/2,b_y+(h-1)/2))
            anchor_list.append(anchor)
    
    return np.array(anchor_list)

def get_target(anchors, gt_boxes):
    bbox_overlap = overlap(anchors, gt_boxes)
    labels = np.zeros((len(anchors)))
    labels.fill(-1)

    anchor_max = np.argmax(bbox_overlap,axis=1) # for a anchor, the ground true that has the biggest covered area
    gt_max = bbox_overlap.argmax(axis=0)#for a gt box, the anchor that has biggest cover area 
    gt_max = np.where(bbox_overlap== bbox_overlap[gt_max,range(bbox_overlap.shape[1])])[0] #change to find the anchor index
    max_overlap_ins = bbox_overlap[range(bbox_overlap.shape[0]),bbox_overlap.argmax(axis=1)]

    labels[max_overlap_ins<RPN_NEGATIVE_OVERLAP] = 0
    labels[max_overlap_ins>RPN_POSITIVE_OVERLAP] = 1
    labels[gt_max] = 1
    fg_num = int(RPN_FG_FACTOR * RPN_SIZE)
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > fg_num:
        remove_index = np.random.choice(fg_index, size=(len(fg_index) - fg_num), replace=False)
        labels[remove_index] = -1
        print "remove:",remove_index 

    bg_num = RPN_SIZE - np.sum(labels==1)
    bg_index = np.where(labels == 0)[0]
    if len(bg_index) > fg_num:
        remove_index = np.random.choice(bg_index, size=(len(bg_index) - bg_num), replace=False)
        labels[remove_index] = -1
    return bbox_overlap, labels

def overlap(anchors, gt_boxes):
    overlaps = np.zeros((len(anchors), len(gt_boxes)))

    for j,g in enumerate(gt_boxes):
        gw = g[2]-g[0]+1
        gh = g[3]-g[1]+1
        gt_area = gw*gh

        for i, a in enumerate(anchors):
            #print i,j
            w = a[2]-a[0]+1
            h = a[3]-a[1]+1

            box_area = w*h

            if a[2] < g[0] or a[0] > g[2]:
                continue
            if a[1] > g[3] or a[3] < g[1]:
                continue
            
            o_x = min(a[2],g[2]) - max(g[0],a[0])+1
            o_y = min(g[3],a[3]) - max(g[1],a[1])+1
            o_s = o_x * o_y
            area = box_area + gt_area - o_s
            overlaps[i, j] = o_s / area

    return overlaps

def get_target_weight(anchors, gt_boxex, bbox, labels):
    max_anchors_index = bbox.argmax(axis=1)
    target = gt_boxex[max_anchors_index,:]

    assert target.shape[0] == anchors.shape[0]

    w = anchors[:,2]-anchors[:,0]+1.0
    h = anchors[:,3]-anchors[:,1]+1.0
    x = anchors[:,0] + w/2
    y = anchors[:,1] + h/2

    t_w = target[:,2]-target[:,0]+1.0
    t_h = target[:,3]-target[:,1]+1.0
    t_x = target[:,0] + t_w/2
    t_y = target[:,1] + t_h/2

    dx = (t_x - x)/w
    dy = (t_y - y)/h
    dw = np.log(t_w/w)
    dh = np.log(t_h/h)

    target_weight = np.vstack( (dx,dy,dw,dh) ).transpose()

    inside_weights = np.zeros( (len(anchors), 4), dtype=np.float32)
    inside_weights[labels == 1, :] = np.array(RPN_BBOX_INSIDE_WEIGHTS)
    outside_weights = np.zeros((len(anchors), 4), dtype=np.float32)


    num_examples = np.sum(labels >= 0) 
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    
    outside_weights[labels == 1, :] = positive_weights
    outside_weights[labels == 0, :] = negative_weights

    return target_weight,inside_weights,outside_weights

def enlarge(data, row, fill_index, default_value):
    tmp = np.empty((row, ) + data.shape[1:], dtype=np.float32)
    tmp.fill(default_value)
    tmp[fill_index] = data
    return tmp


def get_cls_loss(rpn_score, labels):
#calculate coss based on central at images points
    rpn_shape = tf.shape(rpn_score)
    rpn_score = tf.reshape(rpn_score, [-1, 2])
    labels = tf.transpose(labels, [0,3,1,2])
    labels = tf.reshape(labels, [1,1,9*rpn_shape[1],rpn_shape[2]])
    labels= tf.reshape(labels, [-1])

    rpn_cls_score = tf.reshape(tf.gather(rpn_score,tf.where(tf.not_equal(labels,-1))),[-1,2])
    rpn_labels = tf.reshape(tf.gather(labels,tf.where(tf.not_equal(labels,-1))),[-1])
    rpn_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy)

    return rpn_cross_entropy

def get_bbox_loss(bbox_pred, bbox_target, bbox_inside_weight, bbox_outsize_weight):
    loss = tf.multiply(bbox_pred - bbox_target, bbox_inside_weight)
    loss = smooth(loss, sigma=3.0)
    loss = tf.multiply(bbox_outsize_weight,loss)
    loss = tf.reduce_sum(loss)
    loss = RPN_BBOX_LAMBDA * loss
    return loss

def smooth(x, sigma=1.0):
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

if __name__ == '__main__':
    images=np.zeros((1,51,39,18))
    bbox=np.zeros((1,51,39,36))
    #gt_boxes = np.array([[(10,10,20,20,3),(45,45,60,75,4)]])

    gt_boxes = np.array([[ 184.59997559,  203.19999695,  317.3999939 ,  280.,    8. ],
                [ 487.        ,  385.6000061 ,  616.59997559,  524.79998779,    8.        ],
                [ 171.79998779,   91.20000458,  680.59997559,  598.40002441,   15.        ],
                [ 311.44607544,  109.05439758,  439.10574341,  281.05889893,   15.        ],
                [ 458.59054565,  241.4172821 ,  558.03070068,  318.01296997,   15.        ],
                [ 525.10803223,  418.79678345,  607.75073242,  473.89199829,   15.        ]], dtype=np.float32),
    im_info = np.array([[ 800.,  600.,    3.]], dtype=np.float32)

    ik = tf.placeholder(tf.float32, [1,None,None,18])
    gt = tf.placeholder(tf.float32, [1,None,5])
    bbx = tf.placeholder(tf.float32, [None,None,None,36])
    img_size = tf.placeholder(tf.float32, [None,3])

    labels, bbox_target,inside_weights,outside_weights = \
            rpn_anchor_layer(ik, gt, img_size, feat_stride = 16, anchor_scales = [128, 256, 512])

    print labels.shape, np.where(labels==0)
    cls_loss = get_cls_loss(ik, labels)
    bbox_loss = get_bbox_loss(bbx, bbox_target, inside_weights, outside_weights)
    with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            feed_dict_train = {ik:images, gt:gt_boxes, bbx:bbox, img_size:im_info}
            [d]=sess.run([bbox_loss], feed_dict=feed_dict_train)
            print "d:",d*100
            print d.shape
            print np.where(d==0)
            print np.where(d!=0)
            print d[d!=0]*1000
