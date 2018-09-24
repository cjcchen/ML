import tensorflow as tf
import json
import os
import sys
import numpy as np
import collections
import nltk

from dataset.queue import image_queue

def resize_image(image):
    shape = tf.shape(image)
    width = shape[0]
    height = shape[1]
    print width,height

    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def fix_cap(caption):
    caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
    caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
    caption = " ".join(caption.split())  # replace multiple spaces
    return caption.lower()

def word_to_id(captions):
    data = captions
    word_to_id_dict, id_to_word_dict=gen_word_to_id(data)
    data=[ [word_to_id_dict[w] if w in word_to_id_dict else 0 for w in line ] for line in data ]
    return data,word_to_id_dict, id_to_word_dict

def gen_word_to_id(captions):
    if os.path.exists('./data.dat'):
        return get_word_to_id()

    data = np.hstack(captions)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id_dict = dict(zip(words, range(len(words))))
    id_to_word_dict = dict(zip(range(len(words)), words))


    with open("./data.dat","w") as f:
        for w,i in word_to_id_dict.items():
            f.write("%s %d\n" %(w,i))
            if i == 0:
                print w
        f.close()
    return word_to_id_dict,id_to_word_dict


def get_word_to_id():
    if not os.path.exists('./data.dat'):
        return dict(),dict()

    with open("./data.dat","r") as f:
        word_to_id = [line.split() for line in f.readlines()]
        word_to_id_dict = [ (w,int(i)) for w,i in word_to_id]
        id_to_word_dict = [ (int(i),w) for w,i in word_to_id]
        j=0
        for w,i in word_to_id_dict:
            if i <= 1:
                j+=1
        print "single word:",j
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
    for i,c in zip(image_files, captions):
        c = fix_cap(c)
        c=['<START>'] + c.split(' ') + ['<END>']
        if len(c) < max_len+2:
            c+= ['<NULL>'] * (max_len+2-len(c))
        '''
        c=nltk.tokenize.word_tokenize(c.lower())
        c=['<start>'] + c + ['<end>']
        if len(c) > max_len:
                continue
        if len(c) < max_len:
            c+= ['<null>'] * (max_len-len(c))
        image_list.append(i)
        caption_list.append(c)
        #if len(image_list)>100:
        #    break
        c = fix_cap(c)
        '''
        if len(c) > max_len+2:
            continue
        image_list.append(i)
        caption_list.append(c)

    captions, word_to_id_dict, id_to_word_dict=word_to_id(caption_list)
    print ("filter len:",len(captions))
    print ("word list:",len(word_to_id_dict))
    return image_list, captions, word_to_id_dict, id_to_word_dict

def add_queue(image_files, captions, batch_size, mode, threads_num=1):
    data = np.array(captions)
    if mode == 'train':
        image_files, label = tf.train.slice_input_producer([image_files, np.array(captions)],shuffle=True)
    else:
        image_files, label = tf.train.slice_input_producer([image_files, np.array(captions)],shuffle=False)
    image = tf.image.decode_jpeg(tf.read_file(image_files), channels=3)
    #image = resize_image(image)
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    image = tf.cast(image,tf.float32)
    #image = tf.image.per_image_standardization(image)
    data=[image_files, image, label]
    dtype=[image_files.dtype, image.dtype, label.dtype]
    shape=[image_files.shape, image.shape, label.shape]

    return image_queue.data_queue(data=data, dtype=dtype, shape=shape, batch_size=batch_size, mode=mode, threads_num=threads_num)


def get_data(caption_file, image_folder, mode = 'train', max_len = 15, threads_num=1, batch_size=1):
    image_files, captions = read_caption_data(caption_file, image_folder)
    image_files, captions, word_to_id_dict, id_to_word_dict = filter_data(image_files, captions, max_len=max_len)
    image_file, image, label = add_queue(image_files, captions, batch_size, mode, threads_num)
    print label.shape, label.get_shape()[1]-1
    print type(label.get_shape())
    word = tf.slice(input_=label, begin=[0,0],size=[int(label.get_shape()[0]), int(label.get_shape()[1]-1)])
    #word = tf.slice(input_=label, begin=[0,0],size=[tf.shape(label)[0],tf.shape(label)[1]-1])
    #word = tf.slice(input_=label, begin=[0,0],size=[label.get_shape()[0], label.get_shape()[1]-1])
    target = tf.slice(input_=label,begin=[0,1],size=[int(label.get_shape()[0]), int(label.get_shape()[1]-1)])
    return  image_file, image, label,word, target, word_to_id_dict, id_to_word_dict

if __name__ == '__main__':
    tf.flags.DEFINE_string("caption_path", None,
                        "Where the training/test data is stored.")
    tf.flags.DEFINE_string("image_path", None,
                        "Model output directory.")
    FLAGS = tf.flags.FLAGS
    #f,image, c, _,_, w2d, d2w= get_data(sys.argv[1], sys.argv[2])
    f, image,label,word, target, w2d, d2w= get_data(FLAGS.caption_path, FLAGS.image_path, max_len=15, batch_size=1, mode='test')
    from vgg19 import Vgg19
    t_image = tf.placeholder(tf.float32,[None,224,224,3])
    print image
    print t_image
    cnn_net = Vgg19()
    cnn_net.build(image)
    image_feature = cnn_net.conv5_3 #[-1,14*14,512]
    print image.shape
    print ("word len:",len(w2d))
    #f, image,label, word, target, w2d, d2w = get_data(sys.argv[1], sys.argv[2])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess)
        f_,image, c_ = sess.run([f,image, label])
        print "file:",f_
        print image.shape
        print "c:",c_
        print (' '.join([d2w[ll] for ll in c_[0]]))
        '''
        fl,d,l,wd,tg = sess.run([f,image,label, word, target])
        for n,i,c,w,t in zip(fl,d,l,wd, tg):
            print ("cap:",c,w,t,"file:",n)
            print (' '.join([d2w[ll] for ll in c]))
            print (' '.join([d2w[ll] for ll in w]))
            print (' '.join([d2w[ll] for ll in t]))
            print (i.shape, c.shape)
        '''
#_process_caption_data("/home/junechen/ml/annotations/captions_train2014.json", "/home/junechen/ml/",100)
