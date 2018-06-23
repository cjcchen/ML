import numpy as np
import tensorflow as tf
import math

def get_anchors(image, image_raw_size, anchor_ratio, base_anchors):
    anchors = tf.py_func(get_anchors_py,
                                        [image[0], image_raw_size, anchor_ratio, base_anchors],
                                        tf.float32, name="generate_anchors")
    #print ("get anch:",anchors, type(anchors))
    #anchors = tf.convert_to_tensor(tf.cast(anchors,tf.float32), name = 'anchors')
    anchors.set_shape([None, 4])
    return anchors


def get_anchors_py(image, image_raw_size, anchor_ratio, base_anchors):
    #print ("image:",image.shape, "raw:",image_raw_size)

    anchor_size = int(math.ceil(min(image_raw_size[0]/image.shape[0], image_raw_size[1]/image.shape[1])))
    stride_scale = anchor_size
    #print ("an size:",anchor_size, stride_scale)
    anchor_area = anchor_size * anchor_size
    #print ("area:",anchor_area)
    center_x = (anchor_size-1)/2.0
    center_y = (anchor_size-1)/2.0
    #print ("ratio:",anchor_ratio)
    #print ("base:",base_anchors)
    #print ("center:",center_x, center_y)

    #image_scale = np.round(image_raw_size[0] / float(image_h))

    anchor_list = []
    for s in anchor_ratio:
        w = np.round(np.sqrt(anchor_area/s))
        h = np.round(s*w)
        #print ("get w:",w,h)
        for radio in base_anchors:
            rw = w * radio-1
            rh = h * radio-1
            anchor_list.append( [center_x-rw/2.0, center_y-rh/2.0, center_x+rw/2.0,center_y+rh/2.0] )
    res_list = []
    #print ("anchor_list:",anchor_list)

    image_h = image.shape[0]
    image_w = image.shape[1]
    #print ("image h:",image_h, image_w)
    #print ("scale:",image_scale)
    for h in range(image_h):
        for w in range(image_w):
            for anchor in anchor_list:
                res_list.append( [anchor[0]+ w*stride_scale, anchor[1]+ h*stride_scale, anchor[2]+ w*stride_scale, anchor[3]+ h*stride_scale] )
                #res_list.append( [anchor[0] + w, anchor[1] + h, anchor[2] + w, anchor[3] + h] )
    return np.array(res_list, dtype=np.float32)
