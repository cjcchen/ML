import numpy as np


def bbox_inv(anchors, delta_boxes, image_raw_size):
    bbox = np.zeros(delta_boxes.shape, dtype=delta_boxes.dtype)
    ws = anchors[:,2] - anchors[:,0] + 1.0
    hs = anchors[:,3] - anchors[:,1] + 1.0
    cx = anchors[:,0] + ws/2.0
    cy = anchors[:,1] + hs/2.0

    dx = delta_boxes[:,0::4]
    dy = delta_boxes[:,1::4]
    dw = delta_boxes[:,2::4]
    dh = delta_boxes[:,3::4]

    pre_x = dx *ws[:,np.newaxis] + cx[:,np.newaxis]
    pre_y = dy *hs[:,np.newaxis] + cy[:,np.newaxis]
    pre_w = np.exp(dw)*ws[:,np.newaxis]
    pre_h = np.exp(dh)*hs[:,np.newaxis]


    bbox[:,0::4] = pre_x - pre_w/2.0
    bbox[:,1::4] = pre_y - pre_h/2.0
    bbox[:,2::4] = pre_x + pre_w/2.0
    bbox[:,3::4] = pre_y + pre_h/2.0

    bbox[:,0::4] = np.maximum(0, np.minimum(bbox[:,0::4],image_raw_size[1]-1))
    bbox[:,1::4] = np.maximum(0, np.minimum(bbox[:,1::4],image_raw_size[0]-1))
    bbox[:,2::4] = np.maximum(0, np.minimum(bbox[:,2::4],image_raw_size[1]-1))
    bbox[:,3::4] = np.maximum(0, np.minimum(bbox[:,3::4],image_raw_size[0]-1))


    '''
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
    '''

    return bbox

