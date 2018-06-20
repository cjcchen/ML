import numpy as np
import tensorflow as tf

from model.nms_wrapper import nms

def gen_proposal(cls_score_pred, bbox_pred, image_raw_size, anchor_list, num_anchors):
    bois, scores = \
            tf.py_func(get_proposal, [cls_score_pred, bbox_pred, image_raw_size, anchor_list, num_anchors],
                    [tf.float32, tf.float32] )

    bois = tf.convert_to_tensor(tf.cast(bois,tf.float32), name = 'rpn_labels')
    scores = tf.convert_to_tensor(tf.cast(scores,tf.float32), name = 'rois')

    bois.set_shape([None,5])
    scores.set_shape([None,1])
    return bois, scores


def get_proposal(cls_score_pred, bbox_pred, image_raw_size, anchor_list, num_anchors):
    nms_thresh = float(0.7)

    cls_score_pred = cls_score_pred[:,:,:,num_anchors:]
    scores = cls_score_pred.reshape(-1)
    bbox_pred = bbox_pred.reshape(-1,4)
#get the origin box
    bboxes = bbox_inv(anchor_list, bbox_pred, image_raw_size)

#get top n
    bois,scores = get_top_n(bboxes, scores, top = 12000)
#nms
    bois = bois.reshape(-1,4).astype(np.float32)
    scores=scores.reshape(-1,1).astype(np.float32)
    keep = nms(np.hstack((bois, scores)), nms_thresh)


    post_nms_topN = 2000
    keep = keep[:post_nms_topN]

    bois = bois[keep]
    scores = scores[keep]
    old_bois = bois
#get batch size
    zeros = np.zeros((bois.shape[0],1), dtype=np.float32)
    bois = np.hstack((zeros,bois))

    return bois, scores

def bbox_inv(anchors, delta_boxes, image_raw_size):

    bbox = np.zeros((len(anchors), 4))
    for i, (anchor, delta) in enumerate(zip(anchors, delta_boxes)):
        w = anchor[2]-anchor[0]+1.0
        h = anchor[3]-anchor[1]+1.0
        center_x = anchor[0] + w/2
        center_y = anchor[1] + h/2
        w = w.astype(delta_boxes.dtype)
        h = h.astype(delta_boxes.dtype)
        center_x = center_x.astype(delta_boxes.dtype)
        center_y = center_y.astype(delta_boxes.dtype)

        dx = delta[0]
        dy = delta[1]
        dw = delta[2]
        dh = delta[3]

        pre_x = dx*w + center_x
        pre_y = dy*h + center_y
        pre_w = np.exp(dw)*w
        pre_h = np.exp(dh)*h

        pre_x = pre_x.astype(delta_boxes.dtype)
        pre_y = pre_y.astype(delta_boxes.dtype)
        pre_w = pre_w.astype(delta_boxes.dtype)
        pre_h = pre_h.astype(delta_boxes.dtype)
        bbox[i]=(pre_x - pre_w/2.0, pre_y - pre_h/2.0, pre_x + pre_w/2.0, pre_y + pre_h/2.0)
        bbox[i] = bbox[i].astype(delta_boxes.dtype)
        bbox[i][0] = max(0, min(bbox[i][0],image_raw_size[1]-1))
        bbox[i][1] = max(0, min(bbox[i][1],image_raw_size[0]-1))
        bbox[i][2] = max(0, min(image_raw_size[1]-1, bbox[i][2]))
        bbox[i][3] = max(0, min(image_raw_size[0]-1, bbox[i][3]))

    return bbox



def get_top_n(bboxes, scores, top=0):
    idx = scores.argsort()[::-1]
    if top:
        idx = idx[:top]

    bboxes = bboxes[idx]
    scores = scores[idx]

    return bboxes, scores


if __name__ == '__main__':
    cls_scoure_pred = np.zeros([1,2,2,3,2])
    bbox_pred = np.zeros([1, 2,2,3,4])
    for i in range(2):
        for j in range(2):
            for k in range(3):
                bbox_pred[0, i,j,k] = [i,i+k,j,j+k]
                cls_scoure_pred[0,i,j,k] = [0,k/10.0]
    cls_scoure_pred = cls_scoure_pred.reshape(1,2,2,-1).astype(np.float32)
    bbox_pred = bbox_pred.reshape(1,2,2,-1).astype(np.float32)
    print ("cls shape:",cls_scoure_pred.shape, "bbox shape:",bbox_pred.shape)
    print ("cls:",cls_scoure_pred, "bbox:",bbox_pred)
    gen_proposal(cls_scoure_pred, bbox_pred, [4,4,0], [1])




