import tensorflow as tf

from vgg16 import VGG16
from lstm import *


class ImageCaption
    def __init__(self, image, words, target_words, config):
        self.image = image
        
    def build_net(self,images, words):
        self.cnn_net = Vgg16()
        self.cnn_net.build(images)
        self.lstm = LSTM(words, 

        init_state = self.cnn_net.output()
        
        self.lstm = LSTM(initial_state=init_state, input=words, target=target_words, config=config)  



def run_epoch(sess, _input, _target, lstm, lr_decay, epoch_size, summary_writer,sv):
    summary = tf.Summary()
    epochs = 0
    costs = 0
    for e in range(epoch_size):
        cost,lr,summary_str = lstm.train(sess)
        costs += cost
        epochs += config.num_steps
        if e % 10 == 0:
            perxity=np.exp(costs / epochs)
            summary.value.add(tag='perxity', simple_value=perxity)
            summary.value.add(tag='lr', simple_value=lr_decay)
            i_global=sess.run(tf.train.get_or_create_global_step())
            print ("pt %f epo %d global step %d" % (perxity, epochs, i_global))
            summary_writer.add_summary(summary, i_global)#write eval to tensorboard
            summary_writer.add_summary(summary_str, i_global)#write eval to tensorboard
    save_model(sess, sv, i_global)
    return perxity




def main():
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    words = tf.placeholder(tf.int32, [None, None])
    targets = tf.placeholder(tf.int32, [None, None])
    
    c = config

    image_caption = ImageCaption(images, words, targets, c)


    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config_proto) as sess:
        lstm.set_sess(sess)

        print ("log save:",FLAGS.log_path) 
        summary_writer = tf.summary.FileWriter(FLAGS.log_path,sess.graph)

        for i in range(config.max_max_epoch):
            x_lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            print ("lr:",x_lr_decay)
            lstm.assign_lr(config.learning_rate * x_lr_decay)
            p=run_epoch(sess, _input, _target, lstm, x_lr_decay, epoch_size, summary_writer,sv)
            print ("step %d per %f" % (i,p))

