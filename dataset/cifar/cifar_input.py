import os
import re
import numpy
import tensorflow as tf

from dataset.queue import image_queue

def get_shape(image):
    shape = tf.shape(image)
    return shape[0], shape[1]

def get_data(data_files, image_size, channel_num, label_bytes = 1, label_offset = 0, mode = 'train'):
    image_bytes = image_size * image_size * channel_num 
    record_bytes = label_bytes + label_offset + image_bytes

    if mode == 'train':
        file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    else:
        file_queue = tf.train.string_input_producer(data_files, shuffle=False)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])

    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    label = tf.reshape(label, []);
    image = tf.cast(tf.slice(record, [label_offset + label_bytes], [image_bytes]),tf.float32)
    image = tf.reshape(image, [channel_num, image_size, image_size]); 
    image = tf.transpose(image,[1,2,0])
    print image.shape, label.shape
    return image, label

def process_image(image, image_size, mode):

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
# Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
# image = tf.image.random_brightness(image, max_delta=63. / 255.)
# image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
# image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
        print image.shape
        print image_size
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)

    image = tf.image.per_image_standardization(image)
    return image;

def image_input(data_dir, mode='train', image_size = 32, channel_num = 3, batch_size = 20):

    files = []
    if mode =='train':
        for i in range(5):
            files.append( data_dir+'/data_batch_' + str(i + 1)+".bin")
    else:
        files.append(data_dir+'/test_batch.bin')
    image, labels = get_data(files, image_size = image_size, channel_num = channel_num, mode = mode)
    image = process_image(image, image_size, mode) 
    return image_queue.image_queue(image,labels, batch_size = batch_size, mode = mode)


if __name__ == '__main__':
    data_dir="../data/cifar/cifar_10_bin/"
    x,y = image_input(data_dir, 'test', batch_size=20)

    with tf.Session() as sess:
        import numpy as np
# initialize the variables
        sess.run(tf.initialize_all_variables())
# initialize the queue threads to start to shovel data
        threads = tf.train.start_queue_runners(sess)
        
        for i in xrange(10):
            print "from the train set:"
            v_x=sess.run([y])
            print v_x
            #print v_y

        #coord.request_stop()
        #coord.join(threads)
        sess.close()
