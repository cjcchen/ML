
import tensorflow as tf

from scipy.io import loadmat
import numpy as np
import sys
import os

from dataset.queue import image_queue 

def load_label(data_path):

    cache_file = os.path.join(cache_path, 'label.dat')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            labels = cPickle.load(fid)
        return labels

def save_label(data_path,labels):

    cache_file = os.path.join(cache_path, 'label.dat')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(labels, fid, cPickle.HIGHEST_PROTOCOL)

def load_meta(meta_path):
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
            data[wnid] = sid
            words[sid] = word

            print sid, wnid, word

    '''
    ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
    wnids = np.squeeze(np.array([s.WNID for s in synsets]))
    words = np.squeeze(np.array([s.words for s in synsets]))
    print len(wnids), wnids,ids
    print len(words),words,ids
    '''
    return data,words

def load_data_path(data_path):
    path_data, words= load_meta(data_path+"/meta.mat")

    images=[]
    labels=[]
    for image_folder, label in path_data.items():
        image_folder = data_path + image_folder +"/"
        if (os.path.exists(image_folder)):
            files = os.listdir(image_folder)
            if(len(files)==0):
                continue
            print image_folder, label, words[label]
            images+=[ image_folder + f for f in files]
            labels+=[label]*len(files)
    return images, labels, words

def get_data(data_files, labels):
    #print data_files
    #print labels
    file_path, label = tf.train.slice_input_producer([data_files,labels], shuffle=True)
    image = tf.image.decode_jpeg(tf.read_file(file_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def process_image(image, image_size, data_type):
     if data_type == 'train':
         image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
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

def image_input(data_path, batch_size = 2, data_type='train'):
    images, labels, words = load_data_path(data_path)
    image, label = get_data(images, labels)
    image = process_image(image, 224, data_type)
    return image_queue.image_queue(image, label, batch_size, data_type='train')


if __name__ == '__main__':
    data,label = image_input(sys.argv[1], batch_size=10)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess)
        d,label=sess.run([data,label])
        print d.shape, label
