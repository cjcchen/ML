
import tensorflow as tf

from scipy.io import loadmat
import numpy as np
import sys
import os

from dataset.queue import image_queue 

def not_exist(data_path, image_folder):
    image_folder = os.path.join(data_path, image_folder) 
    return not os.path.exists(image_folder)

def load_meta(meta_path, data_path, mode):
    print "load meta:",meta_path
    meta_path = os.path.join(meta_path, "meta.mat")
    metadata = loadmat(meta_path, struct_as_record=False)

  # ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
    synsets = np.squeeze(metadata['synsets'])
    data = {}
    words = {}
    for s in synsets:
        sid = int(np.squeeze(s.ILSVRC2012_ID))
        if sid <1001:
            wnid = str(np.squeeze(s.WNID))
            word = str(np.squeeze(s.words))
            if not_exist(data_path, wnid):
                continue

            data[wnid] = int(sid)-1
            words[int(sid)-1] = word
            #print sid, wnid, word

    if mode == 'eval':
      data = {}
      meta_path = os.path.join(meta_path, "ILSVRC2012_validation_ground_truth.txt")
      with open(meta_path) as fd:
        lines = fd.readlines()
        for p, sid in enumerate(lines):
          if (int(sid) - 1 ) not in words:
            continue
          data[p+1] = int(sid)-1

    return data,words

def load_data_path(meta_path, data_path, mode):
    print "load data path:",data_path
    path_data, words= load_meta(meta_path, data_path, mode)

    images=[]
    labels=[]
    i=0
    for image_folder, label in path_data.items():
        if mode == 'train':
            image_folder = os.path.join(data_path , image_folder) 
            if (os.path.exists(image_folder)):
                images+=[os.path.join(image_folder, p) for p in os.listdir(image_folder) if p.endswith('JPEG')]
                labels+=[label]*len(os.listdir(image_folder))
            else:
                print "folder:",image_folder," not exist"
            i+=1
            assert len(images) == len(labels)

        elif mode == 'eval':
            image_folder = os.path.join(data_path , "ILSVRC2012_val_%08d.JPEG" % int(image_folder))
            if (os.path.exists(image_folder)):
                images+=[image_folder]
                labels+=[label]


    assert len(images) >0
    assert len(images) == len(labels)
    print "data len:",len(images), len(labels)
    #print images,labels
    return images, labels, words

def get_data(data_files, labels):
    file_path, label = tf.train.slice_input_producer([data_files,labels], shuffle=True)
    image = tf.image.decode_jpeg(tf.read_file(file_path), channels=3)
    #image = tf.image.convert_image_dtype(image, tf.float32)
    return file_path, image, label

def get_shape(image):
    shape = tf.shape(image)
    return shape[0], shape[1]

#resize to a image that has a min eage length DEFAULT_MIN_WIDTH
def resize_img(image, resize_min_width):

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



def process_image(image, image_size, data_type):
    if data_type == 'train':
        image = resize_img(image,256)
        #image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        print "image:",image.shape
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        #print image.shape
 # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
 # image = tf.image.random_brightness(image, max_delta=63. / 255.)
 # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
 # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        #image = tf.to_float(image)
        #image = tf.subtract(image, [123.68,116.779,103.939])
        print "image:",image.shape

    else:
        image = resize_img(image,224)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        print image.shape
        print image_size

    image = tf.image.per_image_standardization(image)
    return image;

def image_input(meta_path, data_path, batch_size = 2, mode='train'):
    images, labels, words = load_data_path(meta_path, data_path, mode)
    #print [[images[i], labels[i], words[labels[i]]] for i in xrange(10)]
    file_path, image, label = get_data(images, labels)
    image = process_image(image, 224, mode)
    return image_queue.image_queue(image, label, batch_size, mode=mode), words


if __name__ == '__main__':
    print sys.argv[1], sys.argv[2], sys.argv[3]
    [data,label],words = image_input(sys.argv[1], sys.argv[2], batch_size=10, mode=sys.argv[3])

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      threads = tf.train.start_queue_runners(sess)
      for i in xrange(10):
        d,l = sess.run([data,label])
        for i,ll in enumerate(l):
            print l[i]
            print l[i], words[ll]
        #print d[i].shape, label[i], words[label[i]]
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            imgplot = plt.imshow(d[i])
            plt.show() 
