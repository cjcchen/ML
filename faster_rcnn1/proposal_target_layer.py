import numpy as np
import tensorflow as tf

FG_FRACTION = 0.5
BG_THRESH_HO = 0.5
BG_THRESH_LO = 0.0

def gen_target(proposals, scores, gt_boxes, num_class, batch_size):
    labels, rois, rois_scores, bbox_targets,inside_weights,outside_weights = \
            tf.py_func(get_label, [proposals, scores, gt_boxes, num_class, batch_size],
                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32] )

    labels.set_shape([batch_size, 1])
    rois.set_shape([batch_size, 5])
    rois_scores.set_shape([batch_size])
    bbox_targets.set_shape([batch_size, num_class * 4])
    inside_weights.set_shape([batch_size, num_class * 4])
    outside_weights.set_shape([batch_size, num_class * 4])

    return labels, rois, rois_scores, bbox_targets,inside_weights,outside_weights

def get_label(proposals, scores, gt_boxes, num_class, batch_size):

    zeros = np.zeros((gt_boxes.shape[0], 1), dtype = proposals.dtype)

    all_rois = np.vstack((proposals, np.hstack((zeros, gt_boxes[:, :-1]))))
    label = np.zeros((all_rois.shape[0]), dtype = np.float32)
    #return label, all_rois
    all_scores = np.vstack((scores, zeros))
#class label
    fg_index = []
    bg_index = []
    over_lap_matrix = np.zeros([len(all_rois), len(gt_boxes)])
    for i, anchor in enumerate(all_rois):
        max_area = 0.0
        max_i = 0
        for j,box in enumerate(gt_boxes):
            area = over_lap(anchor[1:5],box[0:4])
            over_lap_matrix[i,j] = area
            if area > max_area:
                max_area = area
                max_i = j
        if max_area >= FG_FRACTION:
            fg_index.append(i)
        elif max_area >= BG_THRESH_LO and max_area < BG_THRESH_HO:
            bg_index.append(i)
        label[i] = gt_boxes[max_i, 4]
    rois_per_image = batch_size #training data
    fg_image_num = np.round(FG_FRACTION * rois_per_image).astype(np.int32) #front ground pages
    if len(fg_index) > 0 and len(bg_index) > 0:
        fg_image_num = min(len(fg_index), fg_image_num)
        bg_image_num = rois_per_image - fg_image_num
        replace = bg_image_num > len(bg_index)

        #print ("fg num:",fg_image_num, "bg num:",bg_image_num, "replace:",replace)
        fg_index = np.random.choice(fg_index, size = fg_image_num, replace = False)
        bg_index = np.random.choice(bg_index, size = bg_image_num, replace = replace)
    elif len(fg_index) > 0:
        replace = rois_per_image > len(fg_index)
        fg_index = np.random.choice(fg_index, size = rois_per_image, replace = replace)
    elif len(bg_index) >0:
        replace = rois_per_image
        bg_index = np.random.choice(bg_index, size = rois_per_image, replace = replace)
    else:
        assert 1 == 0

    final_index = np.append(fg_index, bg_index).astype(np.int32)

    all_rois = all_rois[final_index]
    all_scores = all_scores[final_index]
    label = label[final_index]
    label[len(fg_index):] = 0

#bbox label
    bbox_target = np.zeros( (len(all_rois), 4*num_class), dtype=np.float32)
    in_weight = np.zeros(bbox_target.shape, dtype=np.float32)
    out_weight = np.zeros(bbox_target.shape, dtype=np.float32)
    for i, anchor in enumerate(all_rois):
        cls = int(label[i])
        if cls == 0:
            continue

        anchor = anchor[1:5]
        w = anchor[2]-anchor[0]+1.0
        h = anchor[3]-anchor[1]+1.0
        center_x = anchor[0] + w/2
        center_y = anchor[1] + h/2

        w=w.astype(bbox_target.dtype)
        h=h.astype(bbox_target.dtype)
        center_x=center_x.astype(bbox_target.dtype)
        center_y=center_y.astype(bbox_target.dtype)

        max_gt = over_lap_matrix[final_index[i]].argmax()
        target = gt_boxes[max_gt]

        target_w = target[2]-target[0]+1.0
        target_h = target[3]-target[1]+1.0
        target_center_x = target[0] + target_w/2
        target_center_y = target[1] + target_h/2

        target_w=target_w.astype(bbox_target.dtype)
        target_h=target_h.astype(bbox_target.dtype)
        target_center_x=target_center_x.astype(bbox_target.dtype)
        target_center_y=target_center_y.astype(bbox_target.dtype)

        dx = (target_center_x-center_x)/w
        dy = (target_center_y-center_y)/h
        dw = np.log(target_w/w)
        dh = np.log(target_h/h)

        index = cls * 4
        bbox_target[i, index:index+4] = (dx,dy,dw,dh)
        in_weight[i,index:index+4] = (1,1,1,1)
        out_weight[i,index:index+4] = (1,1,1,1)

    label = label.reshape(-1, 1)
    all_scores = all_scores.reshape(-1)
    return label, all_rois, all_scores, bbox_target,in_weight, out_weight

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
    #print ("over lap:",over_lap_area, total_area, float(over_lap_area)/total_area)
    return float(over_lap_area)/total_area

if __name__ == '__main__':
    proposals = tf.placeholder(tf.float32, [None, 5])
    scores = tf.placeholder(tf.float32, [None, 1])
    gt_box = tf.placeholder(tf.float32, [None, 5])

    #rpn_rois = np.array([[0,0,0,4,4]])
    rpn_rois = np.array([[0,0,0,2,2],[0,10,12,15,18], [0,1,1,4,4]])
    score = np.array([[1],[2]])
    gt_boxes = np.array([[2,2,4,4,1],[11,11,18,18,2]])
    num_class = 3

    #labels, rois =  gen_target(proposals, scores, gt_box, 3, 1)
    labels, rois, rois_scores, bbox_targets,inside_weights,outside_weights =  gen_target(proposals, scores, gt_box, 3, 1)

    sess = tf.Session()
    sess.run(labels, feed_dict={ proposals:rpn_rois, scores: score, gt_box:gt_boxes})
    #blobs = get_label(rpn_rois, gt_boxes,num_class,batch_size=128)
    #print blobs




