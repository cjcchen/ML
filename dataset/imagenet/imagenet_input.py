
import tensorflow as tf

from scipy.io import loadmat
import numpy as np
import sys
import os

from dataset.queue import image_queue 

def load_meta(meta_dir, mode):
    meta_path = os.path.join(meta_dir, "meta.mat")
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
            if sid == 490:
              print sid, word 
            #print sid, wnid, word

    if mode == 'val':
      data = {}
      meta_path = os.path.join(meta_dir, "ILSVRC2012_validation_ground_truth.txt")
      with open(meta_path) as fd:
        lines = fd.readlines()
        for p, sid in enumerate(lines):
          data[p+1] = int(sid)

    return data,words

def load_data_path(meta_path, data_path, mode):
    path_data, words= load_meta(meta_path, mode)

    images=[]
    labels=[]
    for image_folder, label in path_data.items():
        if mode == 'train':
          image_folder = data_path + image_folder +"/"
        elif mode == 'val':
          image_folder = data_path + "ILSVRC2012_val_%08d.JPEG" % int(image_folder)

        if (os.path.exists(image_folder)):
            #print image_folder, label, words[label]
            images+=[image_folder]
            labels+=[label]


    assert len(images) >0
    #print images,labels
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

def image_input(meta_path, data_path, batch_size = 2, mode='train'):
    images, labels, words = load_data_path(meta_path, data_path, mode)
    print [[images[i], labels[i], words[labels[i]]] for i in xrange(10)]
    image, label = get_data(images, labels)
    image = process_image(image, 224, mode)
    return image_queue.image_queue(image, label, batch_size, data_type=mode), words


if __name__ == '__main__':
    with tf.device("/cpu:0"):
      print sys.argv[1], sys.argv[2], sys.argv[3]
      [data,label],words = image_input(sys.argv[1], sys.argv[2], batch_size=10, mode=sys.argv[3])

      with tf.Session() as sess:
          sess.run(tf.initialize_all_variables())
          threads = tf.train.start_queue_runners(sess)
          d,label=sess.run([data,label])
          for i in xrange(len(label)):
            print d[i].shape, label[i], words[label[i]]
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            imgplot = plt.imshow(d[i])
            plt.show() 
