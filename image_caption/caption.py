import tensorflow as tf

from vgg16 import Vgg16
from lstm import lstm
import numpy as np
import data


tf.flags.DEFINE_string("caption_path", None,
                    "Where the training/test data is stored.")
tf.flags.DEFINE_string("image_path", None,
                    "Model output directory.")
FLAGS = lstm.FLAGS


class ImageCaption:
    def __init__(self, image, word, target_word, config):
        self.build_net(image, word, target_word, config)
        self.config = config

    def build_net(self,image, word, target_word, config):
        with tf.name_scope("content_vgg"):
            self.cnn_net = Vgg16()
            self.cnn_net.build(image)

        init_state = self.cnn_net.relu6
        self.lstm = lstm.LSTM(initial_state=init_state, input=word, target=target_word, config=config)


    def run_epoch(self,sess, lr_decay, epoch_size, summary_writer,sv):
        summary = tf.Summary()
        epochs = 0
        costs = 0
        for e in range(epoch_size):
            cost,lr,summary_str = self.lstm.train(sess)
            costs += cost
            epochs += self.config.num_steps
            if e % 10 == 0:
                perxity=np.exp(costs / epochs)
                summary.value.add(tag='perxity', simple_value=perxity)
                summary.value.add(tag='lr', simple_value=lr_decay)
                i_global=sess.run(tf.train.get_or_create_global_step())
                print ("pt %f cost %f epo %d global step %d" % (perxity, costs, epochs, i_global))
                summary_writer.add_summary(summary, i_global)#write eval to tensorboard
                summary_writer.add_summary(summary_str, i_global)#write eval to tensorboard
        save_model(sess, sv, i_global)
        return perxity




def main():
    config = lstm.config()

    f, image,label, word, target, w2d, d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=config.num_steps+1, batch_size=config.batch_size)
    image_caption = ImageCaption(image, word, target, config)
    epoch_size=10000

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config_proto) as sess:
        print ("log save:",FLAGS.log_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_path,sess.graph)

        for i in range(config.max_max_epoch):
            x_lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            print ("lr:",x_lr_decay)
            image_caption.lstm.assign_lr(sess, config.learning_rate * x_lr_decay)
            p=image_caption.run_epoch(sess, x_lr_decay, epoch_size, summary_writer,sv)
            print ("step %d per %f" % (i,p))

if __name__ == '__main__':
    main()
