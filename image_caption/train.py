from solver import CaptioningSolver
from lstm_att_model import Image_Attention
from core.utils import load_coco_data
import numpy as np

import os
import data
from vgg19 import Vgg19
import tensorflow as tf
import utils
from PIL import Image
from scipy import ndimage
import skimage
import skimage.io
import skimage.transform

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class img_config():
    hidden_size=1024
    embedding_size=512
    feature_dim = [196,512]
    seq_len = 16
    learning_rate=0.001


path="/home/tusimple/junechen/ml_data/data/show"
log_path='%s/model_log1' % path
model_path='%s/model_model1' % path


def train():
    config = img_config()
    config.batch_size=128
    caption_path = '/home/tusimple/junechen/ml_data/data/annotations/captions_train2014.json'
    image_path = '/home/tusimple/junechen/ml_data/data/train2014'
    f, image,label,word, target, w2d,d2w = data.get_data(caption_path, image_path, max_len=15, batch_size=config.batch_size)
    model = Image_Attention(w2d=w2d, config=config)

    with tf.name_scope("content_vgg"):
        cnn_net = Vgg19()
        cnn_net.build(image)
        image_feature = cnn_net.conv5_3 #[-1,14*14,512]

    n_iters_per_epoch = int(np.ceil(float(400000)/config.batch_size))

    # build graphs for training model and sampling captions
    # This scope fixed things!!
    with tf.variable_scope(tf.get_variable_scope()):
        loss = model.build()
    i_global_op=tf.train.get_or_create_global_step()

    # train op
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=tf.train.get_or_create_global_step())

    # summary op
    # tf.scalar_summary('batch_loss', loss)
    tf.summary.scalar('batch_loss', loss)
    for var in tf.trainable_variables():
        #tf.histogram_summary(var.op.name, var)
        tf.summary.histogram(var.op.name, var)
    for grad, var in grads_and_vars:
        print grad, var
        #tf.histogram_summary(var.op.name+'/gradient', grad)
        tf.summary.histogram(var.op.name+'/gradient', grad)

    #summary_op = tf.merge_all_summaries()
    summary_op = tf.summary.merge_all()


    #sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    #with sv.managed_session(config=config_proto) as sess:
    with tf.Session(config=config_proto) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        threads = tf.train.start_queue_runners(sess)

        summary_writer = tf.summary.FileWriter(log_path,sess.graph)

        curr_loss = 0.0
        for e in range(20):
            for i in range(n_iters_per_epoch):
                features_batch, captions_batch = sess.run([image_feature,label])
                features_batch = features_batch.reshape([-1,14*14,512])
                word = captions_batch[:,0:-1]
                target = captions_batch[:,1:]
                feed_dict = {model.feature: features_batch, model.word:word, model.target:target}
                _, l,gs = sess.run([train_op, loss,i_global_op], feed_dict)
                if i % 10 == 0:
                    print "\nTrain loss at global step %d epoch %d & iteration %d (mini-batch): %.5f" %(gs, e+1, i+1, l)

def get_train_op(loss):
    # train op
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=tf.train.get_or_create_global_step())
    return train_op

def resize_image(img):
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224,3))
    return resized_img


def read_image(files):
    img = []
    for f in files:
        image=Image.open(f)
        image = ndimage.imread(f, mode='RGB')
        #print type(image)
        #print f,image.shape
        image = resize_image(image)
        #print f,image.shape
        #image = utils.load_image(f)
        image = image.reshape(224, 224, 3)
        img.append(image)
    return np.array(img).reshape(-1,224,224,3)

def base_train1():

    config = img_config()
    config.batch_size=128

    caption_path = '/home/tusimple/junechen/ml_data/data/annotations/captions_train2014.json'
    image_path = '/home/tusimple/junechen/ml_data/data/train2014'
    #f, image,label,word, target, w2d,d2w = data.get_data(caption_path, image_path, max_len=15, batch_size=config.batch_size)


    image_files, captions = data.read_caption_data(caption_path, image_path)
    image_files, captions, word_to_id_dict, id_to_word_dict = data.filter_data(image_files, captions, max_len=15)

    model = Image_Attention(w2d=word_to_id_dict, config=config)

    t_image = tf.placeholder(tf.float32, [None,224,224,3])
    with tf.name_scope("content_vgg"):
        cnn_net = Vgg19()
        cnn_net.build(t_image)
        image_feature = cnn_net.conv5_3 #[-1,14*14,512]


    word_to_idx = word_to_id_dict
    idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    model = Image_Attention(w2d=word_to_idx, config=config)
    n_examples=400000-2
    #n_examples = data['features'].shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples)/config.batch_size))

    # build graphs for training model and sampling captions
    # This scope fixed things!!
    with tf.variable_scope(tf.get_variable_scope()):
        loss = model.build()
    i_global_op=tf.train.get_or_create_global_step()

    train_op=get_train_op(loss)

#    optimizer = tf.train.AdamOptimizer
    image_files = np.array(image_files)
    n_epochs=20
    n_examples = image_files.shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples)/config.batch_size))
    captions_data = np.array(captions)
    print captions[0]

    #config.gpu_options.per_process_gpu_memory_fraction=0.9
    t=tf.ConfigProto(allow_soft_placement = True)
    #t.gpu_options.allow_growth = True
    with tf.Session(config=t) as sess:
        tf.global_variables_initializer().run()
        for e in range(n_epochs):
            rand_idxs = np.random.permutation(n_examples)
            captions = captions_data[rand_idxs]
            image_f = image_files[rand_idxs]

            for i in range(n_iters_per_epoch):
                captions_batch = captions[i*config.batch_size:(i+1)*config.batch_size]
                image_f_batch = image_f[i*config.batch_size:(i+1)*config.batch_size]
                image_batch = read_image(image_f_batch)

                features_batch = sess.run([image_feature], feed_dict={t_image:image_batch})
                features_batch = features_batch[0].reshape(-1,14*14,512)
                word = captions_batch[:,0:-1]
                target = captions_batch[:,1:]
                feed_dict = {model.feature: features_batch, model.word:word, model.target:target}
                _, l,gs = sess.run([train_op, loss,i_global_op], feed_dict)

                # write summary for tensorboard visualization
                if i % 10 == 0:
                    print "\nTrain loss at global step %d epoch %d & iteration %d (mini-batch): %.5f" %(gs, e+1, i+1, l)


def base_train():

    config = img_config()
    config.batch_size=128
    data = load_coco_data(data_path=path, split='train')

    word_to_idx = data['word_to_idx']
    idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    model = Image_Attention(w2d=word_to_idx, config=config)
    #n_examples=400000-2
    n_examples = data['features'].shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples)/config.batch_size))

    # build graphs for training model and sampling captions
    # This scope fixed things!!
    with tf.variable_scope(tf.get_variable_scope()):
        loss = model.build()
    i_global_op=tf.train.get_or_create_global_step()

    train_op=get_train_op(loss)



#    optimizer = tf.train.AdamOptimizer

    n_epochs=20
    n_examples = data['features'].shape[0]
    n_iters_per_epoch = int(np.ceil(float(n_examples)/config.batch_size))
    features = data['features']
    captions_data = data['captions']
    image_idxs_data = data['image_idxs']

    #config.gpu_options.per_process_gpu_memory_fraction=0.9
    t=tf.ConfigProto(allow_soft_placement = True)
    #t.gpu_options.allow_growth = True
    with tf.Session(config=t) as sess:
        tf.global_variables_initializer().run()
        for e in range(n_epochs):
            rand_idxs = np.random.permutation(n_examples)
            captions = captions_data[rand_idxs]
            image_idxs = image_idxs_data[rand_idxs]

            for i in range(n_iters_per_epoch):
                captions_batch = captions[i*config.batch_size:(i+1)*config.batch_size]
                image_idxs_batch = image_idxs[i*config.batch_size:(i+1)*config.batch_size]
                features_batch = features[image_idxs_batch]
                word = captions_batch[:,0:-1]
                target = captions_batch[:,1:]
                feed_dict = {model.feature: features_batch, model.word:word, model.target:target}
                _, l,gs = sess.run([train_op, loss,i_global_op], feed_dict)

                # write summary for tensorboard visualization
                if i % 10 == 0:
                    print "\nTrain loss at global step %d epoch %d & iteration %d (mini-batch): %.5f" %(gs, e+1, i+1, l)


def train_base():
    config = img_config()
    data = load_coco_data(data_path=path, split='train')
    print "load coco data done"
    word_to_idx = data['word_to_idx']
    #load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path=path, split='val')

    model = Image_Attention(w2d=word_to_idx, config=config)



    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=100, save_every=1, image_path='%s/'%path,
                                    pretrained_model=None, model_path='%s/model5/lstm/'%path, test_model='%s/model5/lstm/model-10'%path,
                                     print_bleu=True, log_path='%s/log5/'%path)

    solver.train()


def main():
    #train()
    #train_base()
    #train_data()
    base_train1()
    '''
    config = img_config()

    # load train dataset
    data = load_coco_data(data_path=path, split='train')

    word_to_idx = data['word_to_idx']
    idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    print data['file_names'][0:10]
    print data['captions'][0:10]
    print "word_to_idx:",len(word_to_idx)

    for data in data['captions'][0:10]:
        for x in data:
            print idx_to_word[x]
    '''

    #data = dict({'word_to_idx':{'<START>':0,'<NULL>':1},'features':numpy.array([123]),'captions':numpy.array([123]),'image_idxs':numpy.array([123])})
    #print "load coco data done"
    # load val dataset to print out bleu scores every epoch
    #val_data = load_coco_data(data_path=path, split='val')

    #model = Image_Attention(w2d=word_to_idx, config=config)



    #solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
    #                                      learning_rate=0.001, print_every=100, save_every=1, image_path='%s/'%path,
    #                                pretrained_model=None, model_path='%s/model5/lstm/'%path, test_model='%s/model5/lstm/model-10'%path,
    #                                 print_bleu=True, log_path='%s/log5/'%path)

    #solver.train()

if __name__ == "__main__":
    main()
