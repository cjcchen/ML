import time
import os
import numpy as np
import tensorflow as tf
import resnet 
from dataset.cifar import cifar_input
from component import network_train

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/img_train/', 'data dir.')
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_string('cpu_mode', 'cpu', "cpu? or gpu?")
tf.app.flags.DEFINE_string('run_mode', 'train', "train or eval")

def eval():

    test_x,test_y=cifar_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size,mode='test')

    with tf.device("/"+FLAGS.cpu_mode+":0"):
        #model = resnet_model.ResNet('train')
        images=tf.placeholder(tf.float32, [None,32,32,3])
        labels=tf.placeholder(tf.int32, [None]) 
        with tf.variable_scope("resnet", reuse=None):
          eval_model = resnet.ResNet(images, labels, num_class=10, mode='eval')

    save_dir = FLAGS.save_dir
    save_path = save_dir+"resnet"
    log_dir=FLAGS.log_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saver = tf.train.Saver()
    session = network_train.load_session(save_dir, saver)
    threads = tf.train.start_queue_runners(session)

    eval_account = 0
    correct_account = 0
    while eval_account<10000:
            t_x,t_y = session.run([test_x,test_y])

            t_acc_count, t_acc = network_train.eval_predict(session, eval_model, t_x,t_y)

            correct_account += t_acc_count
            eval_account += len(t_x)

            print "eval account:", eval_account,"test acc:",t_acc, "total:", float(correct_account)/eval_account*100


def main(argv=None):  

    if FLAGS.run_mode=="eval":
      eval()
    else:
      x,y=cifar_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size)
      test_x,test_y=cifar_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size,data_type='test')

      with tf.device("/"+FLAGS.cpu_mode+":0"):
          #model = resnet_model.ResNet('train')
          with tf.variable_scope("resnet", reuse=None):
              train_x=tf.placeholder(tf.float32, [None,32,32,3])
              train_y=tf.placeholder(tf.int32, [None]) 
              model = resnet.ResNet(train_x,train_y,num_class=10, mode='train')

          with tf.variable_scope("resnet", reuse=True):
              e_x=tf.placeholder(tf.float32, [None,32,32,3])
              e_y=tf.placeholder(tf.int32, [None]) 
              eval_model = resnet.ResNet(e_x,e_y,num_class=10, mode='eval')
      network_train.train(model,eval_model, x,y,test_x,test_y)

if __name__ == '__main__':
    tf.app.run()
