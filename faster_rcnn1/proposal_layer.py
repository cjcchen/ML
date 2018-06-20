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
    print("get proposal, score:",cls_score_pred.shape,"bbox:",bbox_pred.shape)
    print("num anchor_list:",num_anchors)
    print ("image_raw_size:",image_raw_size)
    nms_thresh = float(0.7)

    #get the front pages
    cls_score_pred = cls_score_pred[:,:,:,num_anchors:]
    #print("score shape:",cls_score_pred.shape)
    scores = cls_score_pred.reshape(-1)
    #print("score shape:",scores.shape)

    #get the points for each front pages
    bbox_pred = bbox_pred.reshape(-1,4)
    #print ("box pred shape:",bbox_pred.shape)
    #print ("score:",scores)
    #print ("box:",bbox_pred)
#get the origin box
    bboxes = bbox_inv(anchor_list, bbox_pred, image_raw_size)
    #print ("inv:",bboxes, bboxes.shape)

#get top n
    bois,scores = get_top_n(bboxes, scores, top = 12000)
    print ("== = = = = after top:%.10f %.10f %.10f %.10f"%(bois[0][0], bois[0][1], bois[0][2], bois[0][3]))
    #print ("top n, bois:",bois)
    #print ("top n, scores:",scores)
    #print ("shape:",bois.shape, scores.shape)
    #print (type(bois), type(scores))
#nms
    bois = bois.reshape(-1,4).astype(np.float32)
    scores=scores.reshape(-1,1).astype(np.float32)
    keep = nms(np.hstack((bois, scores)), nms_thresh)


    post_nms_topN = 2000
    keep = keep[:post_nms_topN]

    bois = bois[keep]
    scores = scores[keep]
    old_bois = bois
    #print ("nms n, bois:",bois)
    #print ("nms n, scores:",scores)

#get batch size
    zeros = np.zeros((bois.shape[0],1), dtype=np.float32)
    bois = np.hstack((zeros,bois))

    print("get res score:",scores,"bbox:",bois)

    #print ("final top n, bois:",bois)
    #print ("final top n, scores:",scores, scores.shape)
    return bois, scores

def bbox_inv(anchors, delta_boxes, image_raw_size):

    #print ("delta box:",delta_boxes)
    print("inv:",anchors.dtype,delta_boxes.dtype)
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
        if i == 8754:
            print ("??? type:",pre_x.dtype, pre_y.dtype, pre_w.dtype, pre_h.dtype, (pre_x-0.5*pre_w).dtype)
            print ("pre:%.10f %.10f %.10f %.10f( %.10f, %.10f, %.10f)"%(pre_x, pre_y, pre_w, pre_h, pre_x-pre_w/2,pre_x-0.5*pre_w, pre_x-pre_w/2.0 ))
        bbox[i]=(pre_x - pre_w/2.0, pre_y - pre_h/2.0, pre_x + pre_w/2.0, pre_y + pre_h/2.0)
        bbox[i] = bbox[i].astype(delta_boxes.dtype)
        if i == 8754:
            print ("???%.10f" % bbox[i][0])
            print ("xxx:",w,h,center_x, center_y)
            print ("dx,dy:",dx,dy,dw,dh)
            print ("%.10f %.10f %.10f %.10f, %.10f"%(pre_x, dx,w,center_x, dx*w),type(dx),type(w))
            print ("pred:",pre_x, pre_y, pre_w, pre_h)
            print ("pro box:",bbox[0])
        if i == 8754:
            print ("1111 == = = = =:%.10f %.10f %.10f %.10f"%(bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]))
            print ("img:",image_raw_size[0], image_raw_size[1])
        bbox[i][0] = max(0, min(bbox[i][0],image_raw_size[1]-1))
        bbox[i][1] = max(0, min(bbox[i][1],image_raw_size[0]-1))
        bbox[i][2] = max(0, min(image_raw_size[1]-1, bbox[i][2]))
        bbox[i][3] = max(0, min(image_raw_size[0]-1, bbox[i][3]))
        if i == 8754:
            print ("== = = = =:%.10f %.10f %.10f %.10f"%(bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]))

    return bbox



def get_top_n(bboxes, scores, top=0):
    idx = scores.argsort()[::-1]
    print ("top n oder:",idx[0:10])
    #print ("top n bbox shape:",bboxes.shape, "scores shape:",scores.shape)
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




