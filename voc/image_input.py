import os
import re
import tensorflow as tf

id=0
index_list={}

def get_label_index(name):
    global id
    global index_list
    if name not in index_list:
        index_list[name]=id
        id+=1

    return index_list[name]

def get_data_files(data_dir):

    path=path1=os.path.abspath(data_dir)+"/"
    files = os.listdir(data_dir)
   
    img_files=[]
    img_labels=[]

    p={}
    for img_fn in files:
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = re.search(r'(n\d+)', img_fn).group(1)

        if label_name not in p:
            p[label_name]=1
        else:
            p[label_name]=p[label_name]+1

        if p[label_name] > 20:
            continue

        print "filename:",img_fn,"label name:",get_label_index(label_name)


        img_files.append(path+img_fn)
        img_labels.append(get_label_index(label_name))

    print len(img_files), len(img_labels)
    return img_files, img_labels

IMAGE_HEIGHT=224
IMAGE_WIDTH=224
NUM_CHANNELS=3
BATCH_SIZE=64
_R_MEAN = 123.68 / 255
_G_MEAN = 116.78 / 255
_B_MEAN = 103.94 / 255
DEFAULT_MIN_WIDTH = 256

def get_shape(image):
    shape = tf.shape(image)
    return shape[0], shape[1]

#resize to a image that has a min eage length DEFAULT_MIN_WIDTH
def resize_img(image, resize_min_width = DEFAULT_MIN_WIDTH ):

    h,w = get_shape(image)

    resize_min_width = tf.cast(resize_min_width, tf.float32)
    height = tf.cast(h, tf.float32)
    width = tf.cast(w, tf.float32)

    small_dim = tf.minimum(height, width)

    scale_ratio = resize_min_width / small_dim
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    resized_image = tf.image.resize_images(image, 
            [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR, 
            align_corners=False)

    return resized_image

def random_crop_and_flip(image, crop_height, crop_width):

    height, width = get_shape(image)

    total_crop_height = (height - crop_height)
    crop_top = tf.random_uniform([], maxval=total_crop_height + 1, dtype=tf.int32)
    total_crop_width = (width - crop_width)
    crop_left = tf.random_uniform([], maxval=total_crop_width + 1, dtype=tf.int32)

    cropped = tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

    cropped = tf.image.random_flip_left_right(cropped)
    return cropped

def mean_image_subtraction(image, means):
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    return image - means


def get_data(data_dir):

    img_files, img_labels= get_data_files(data_dir)
    filename, label = tf.train.slice_input_producer([img_files, img_labels], shuffle=True)
    image = tf.image.decode_jpeg(tf.read_file(filename), channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def process_image(image):
    image = resize_img(image);
    image = random_crop_and_flip(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    image = tf.cast(image, tf.float32)

    return mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

def input_queue(image, label, batch_size, threads_num = 1):

    height, width = get_shape(image)
    num_channels = image.get_shape().as_list()[-1]

    example_queue = tf.RandomShuffleQueue(
    capacity= (threads_num +2)* batch_size,
            min_after_dequeue= batch_size,
            dtypes=[tf.float32,tf.int32],
            shapes=[image.shape, label.shape])
            #shapes=[[height, width, num_channels], []])

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * threads_num))
    image, label = example_queue.dequeue_many(batch_size)
    return  image, label


def image_input(data_dir):
    image, labels = get_data(data_dir)
    image = process_image(image) 
    return input_queue(image,labels, batch_size = BATCH_SIZE)



if __name__ == '__main__':
    data_dir="data/img_train/"
    x,y = image_input(data_dir)
    print x

    with tf.Session() as sess:
# initialize the variables
        sess.run(tf.initialize_all_variables())

# initialize the queue threads to start to shovel data
        #coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess)

        for i in xrange(10):
            print "from the train set:"
            v_x,v_y=sess.run([x,y])
            print v_x
            print v_y

        #coord.request_stop()
        #coord.join(threads)
        sess.close()
