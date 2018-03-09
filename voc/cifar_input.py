import os
import re
import numpy
import tensorflow as tf

def get_shape(image):
    shape = tf.shape(image)
    return shape[0], shape[1]

def get_data(data_files, image_size, channel_num, label_bytes = 1, label_offset = 0):
    print data_files
    image_bytes = image_size * image_size * channel_num 
    record_bytes = label_bytes + label_offset + image_bytes

    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
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

def process_image(image, image_size, data_type):

    if data_type == 'train':
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

def input_queue(image, label, batch_size, threads_num = 1, data_type='train'):
    print batch_size

    height, width = get_shape(image)
    num_channels = image.get_shape().as_list()[-1]

    if data_type == 'train':
        example_queue = tf.RandomShuffleQueue(
        capacity= (threads_num +2)* batch_size,
                min_after_dequeue= batch_size,
                dtypes=[tf.float32,tf.int32],
                shapes=[image.shape, []])
    else:
        threads_num = 1
        example_queue = tf.FIFOQueue(
                (threads_num + 3) * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[image.shape, []])

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * threads_num))
    image, label = example_queue.dequeue_many(batch_size)
    return  image, label


def image_input(data_dir, data_type='train', image_size = 32, channel_num = 3, batch_size = 1):

    files = []
    if data_type=='train':
        for i in range(5):
            files.append( data_dir+'/data_batch_' + str(i + 1)+".bin")
    else:
        files.append(data_dir+'/test_batch.bin')
    image, labels = get_data(files, image_size = image_size, channel_num = channel_num)
    image = process_image(image, image_size, data_type) 
    return input_queue(image,labels, batch_size = batch_size)



if __name__ == '__main__':
    data_dir="data/cifar_10_bin/"
    #x,y = get_data([data_dir+"/data_batch_1.bin"],32,3,1)
    #x,y = get_data([data_dir+"/data_batch_1"],32,3,1)
    x,y = image_input(data_dir, 'test')
    print x

    with tf.Session() as sess:
        import resnet
        import numpy as np
# initialize the variables
        mean = tf.reduce_mean(x)
        res = resnet.bn(x)
        mean1 = tf.reduce_mean(res)
        sess.run(tf.initialize_all_variables())

# initialize the queue threads to start to shovel data
        #coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess)
        

        for i in xrange(10):
            print "from the train set:"
            v_x,m,r=sess.run([x,mean,mean1], {resnet.is_training:False})
            print m
            print r
            print m>r
            print len(v_x)
            image_np = v_x[0]
            print "len:",len(image_np), image_np[3,...]
            for i in range(len(image_np)):
                print i
                meanx = np.mean(image_np[i, ...])
                print "mean 1:",meanx
                std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(32* 32* 3)])
                image_np[i,...]=(image_np[i, ...] - meanx) / std
            print len(image_np[0])
                
            imp = image_np
            for i in range(len(imp)):
                meanx = np.mean(imp[i, ...])
                print "mean 2:",meanx

            #print image_np
            #v_x,v_y=sess.run([x,y])
            #print v_x
            #print v_y

        #coord.request_stop()
        #coord.join(threads)
        sess.close()
