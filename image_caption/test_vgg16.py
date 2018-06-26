import numpy as np
import tensorflow as tf

import vgg16
import utils
import data
import sys

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [None, 224, 224, 3])

    f, image,label, w2d, d2w = data.get_data(sys.argv[1], sys.argv[2], batch_size=1)

    sess.run(tf.initialize_all_variables())
    threads = tf.train.start_queue_runners(sess)

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(image)

    d,l = sess.run([image,label])
    print ([d2w[ll] for ll in l[0]])
    feed_dict = {images: d}

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    #print(prob)
    utils.print_prob(prob[0], './synset.txt')
