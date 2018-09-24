import tensorflow as tf
import json
import os
import sys
import numpy as np
import collections
import nltk

from dataset.queue import image_queue


word_to_id_dict={}

def word_to_id(captions):
    data = captions
    word_to_id_dict, id_to_word_dict=gen_word_to_id(data)
    data=[ [word_to_id_dict[w] for w in line] for line in data ]
    return data,word_to_id_dict, id_to_word_dict

def gen_word_to_id(captions):
    data = np.hstack(captions)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id_dict = dict(zip(words, range(len(words))))
    id_to_word_dict = dict(zip(range(len(words)), words))

    with open("./data.dat","w") as f:
        for w,i in word_to_id_dict.items():
            print ("%s %d"%(w,i))
            f.write("%s %d\n" %(w,i))
        f.close()

    return word_to_id_dict,id_to_word_dict

def get_word_to_id():
    if not os.path.exists('./data.dat'):
        return dict(),dict()

    with open("./data.dat","r") as f:
        word_to_id = [line.split() for line in f.readlines()]
        word_to_id_dict = [ (w,int(i)) for w,i in word_to_id]
        id_to_word_dict = [ (int(i),w) for w,i in word_to_id]
        return dict(word_to_id_dict), dict(id_to_word_dict)

def read_caption_data(caption_file, image_folder):
    with open(caption_file) as f:
        caption_data = json.load(f)
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    image_files=[]
    captions=[]
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        image_files.append(os.path.join(image_folder, id_to_filename[image_id]))
        captions.append(annotation['caption'])
    print ("read file:",len(image_files))
    return image_files, captions

def filter_data(image_files, captions, max_len):
    image_list = []
    caption_list = []
    rec = []
    for i,c in zip(image_files, captions):
        rec.append(c)
        c=nltk.tokenize.word_tokenize(c.lower())
        c=['<start>'] + c + ['<end>']
        image_list.append(i)
        caption_list.append(c)

        if len(image_list) > 20:
            break

    xc, word_to_id_dict, id_to_word_dict=word_to_id(caption_list)
    print ("captions:",rec)
    print ("rest:",xc)
    caption_list = [ ' '.join(c) for c in caption_list ]
    print ("filter len:",len(caption_list))
    return image_list, rec, word_to_id_dict, id_to_word_dict


def add_queue(image_files, captions, batch_size, mode, threads_num=1):
    data = np.array(captions)
    image_files, label = tf.train.slice_input_producer([image_files, np.array(captions)],shuffle=True)
    image = tf.image.decode_jpeg(tf.read_file(image_files), channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    image = tf.image.per_image_standardization(image)

    data=[image_files, image, label]
    dtype=[image_files.dtype, image.dtype, label.dtype]
    shape=[image_files.shape, image.shape, label.shape]

    return image_queue.data_queue(data=data, dtype=dtype, shape=shape, batch_size=batch_size, mode=mode, threads_num=threads_num)

def caption_py(captions):
    w2d=word_to_id_dict
    print ("get capt:",captions)
    #w2d, d2w = get_word_to_id()
    print ("list len:",len(w2d))
    caption_list = []
    for c in captions:
        c=nltk.tokenize.word_tokenize(c.lower())
        c=['<start>'] + c + ['<end>']
        caption_list.append(c)

    data=[ [w2d[w] for w in line] for line in caption_list]
    data_len=[ len(line) for line in caption_list]
    print ("ret:",data)
    return data,data_len


def get_data(caption_file, image_folder, mode = 'train', max_len = 15, threads_num=1, batch_size=2):
    image_files, captions = read_caption_data(caption_file, image_folder)

    image_files, captions, w2d, d2w  = filter_data(image_files, captions, max_len=max_len)
    global word_to_id_dict
    word_to_id_dict = w2d

    image_file, image, label = add_queue(image_files, captions, batch_size, mode, threads_num)
    print ("xxx:",len(w2d))
    return  image_file, image, label, w2d, d2w

if __name__ == '__main__':
    f, image,label,w2d,d2w = get_data(sys.argv[1], sys.argv[2])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess)
        #for i in range(2):
        fl,d,l = sess.run([f,image,label])
        for n,i,c in zip(fl,d,l):
            print ("cap:",c,"file:",n)
        print (caption_py(l))
            #print (' '.join([d2w[ll] for ll in c]))
            #print (' '.join([d2w[ll] for ll in w]))
            #print (' '.join([d2w[ll] for ll in t]))
            #print (i.shape, c.shape)
        #[d]= sess.run([image])
        #import matplotlib.pyplot as plt
        #import matplotlib.image as mpimg
        #imgplot = plt.imshow(d)
        #plt.show()
#_process_caption_data("/home/junechen/ml/annotations/captions_train2014.json", "/home/junechen/ml/",100)
