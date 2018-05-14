import tensorflow as tf

from vgg16 import Vgg16
from lstm import lstm
import numpy as np
import data
import utils
import sys


tf.flags.DEFINE_string("caption_path", None,
                    "Where the training/test data is stored.")
tf.flags.DEFINE_string("image_path", None,
                    "Model output directory.")
FLAGS = lstm.FLAGS

class ImageCaption:
    def __init__(self, images, word, target, config, is_training=True):
        self.config = config

        print ("is train:",is_training)
        self.build_net(images, word, target, config, is_training=is_training)

    def build_net(self,image, word, target_word, config, is_training=True):
        with tf.name_scope("content_vgg"):
            self.cnn_net = Vgg16()
            self.cnn_net.build(image)

        with tf.device("/gpu:0"):
            init_state = self.cnn_net.relu6
            self.lstm = lstm.LSTM(initial_state=init_state, input=word, target=target_word, config=config, is_training=is_training)

    def run_epoch(self,sess, lr_decay, epoch_size, summary_writer,sv):
        summary = tf.Summary()
        epochs = 0
        costs = 0
        for e in range(epoch_size):
            feed_dict={}
            cost,lr,summary_str = self.lstm.train(sess, feed_dict)
            costs += cost
            epochs += self.config.num_steps
            if e % 10 == 0:
                perxity=np.exp(costs / epochs)
                summary.value.add(tag='perxity', simple_value=perxity)
                summary.value.add(tag='lr', simple_value=lr_decay)
                i_global=sess.run(tf.train.get_or_create_global_step())
                print ("pt %f cost %f pre cost %f epo %d global step %d" % (perxity, costs, cost, epochs, i_global))
                summary_writer.add_summary(summary, i_global)#write eval to tensorboard
                summary_writer.add_summary(summary_str, i_global)#write eval to tensorboard
            if e % 100 == 0:
                save_model(sess, sv, FLAGS.save_path, i_global)
        return perxity

def save_model(sess, sv, save_path, global_step):
    sv.save(sess, save_path=save_path, global_step=global_step)

def load_session(sess, checkpoint_dir):
    saver = tf.train.Saver()

    try:
        print("Trying to restore last checkpoint ...:",checkpoint_dir)
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        saver.restore(sess, save_path=last_chk_path)
        print("restore last checkpoint %s done"%checkpoint_dir)
    except Exception as e:
        print("Failed to restore checkpoint. Initializing variables instead."),e
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

    return saver

def train():
    config = lstm.config()
    config.batch_size = 2
    config.hidden_size=512

    f, image,label,word, target, w2d,d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=config.num_steps+1, batch_size=config.batch_size)
    epoch_size=10000
    config.vocab_size = len(w2d)
    print ("vb size:",len(w2d))
    image_caption = ImageCaption(image, word, target, config)

    #sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    #with sv.managed_session(config=config_proto) as sess:
    with tf.Session(config=config_proto) as sess:
        sv = load_session(sess, FLAGS.save_path)
        threads = tf.train.start_queue_runners(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.log_path,sess.graph)

        for i in range(config.max_max_epoch):
            x_lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            print ("lr:",x_lr_decay)
            image_caption.lstm.assign_lr(sess, config.learning_rate * x_lr_decay)
            p=image_caption.run_epoch(sess, x_lr_decay, epoch_size, summary_writer,sv)
            print ("step %d per %f" % (i,p))

def predict():
    config = lstm.config()
    config.hidden_size=512
    config.batch_size=1

    img = utils.load_image("./test_data/tiger.jpeg")
    img = img.reshape((1, 224, 224, 3))
    w2d, d2w = data.get_word_to_id()
    print "read w2d size:",len(w2d)
    if len(w2d) == 0:
        f, image,label, word, target, w2d, d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=config.num_steps+1, batch_size=config.batch_size)
        print "reload read w2d size:",len(w2d)
    config.vocab_size = len(w2d)
    #config.vocab_size = 24553

    images = tf.placeholder("float", [None, 224, 224, 3], name="image")
    word = tf.placeholder(tf.int32, [None, None], name="word_seq")
    image_caption = ImageCaption(images, word, None, config, is_training=False)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config_proto) as sess:
        seq = [w2d['<start>']]
        for i in range(50):
            seq_len = np.array(len(seq)).reshape([-1])
            feed_dict = {
                    images:img,
                    word:np.array(seq).reshape([1,-1]),
                    }

            output = sess.run(image_caption.lstm.logits, feed_dict=feed_dict)
            print (output.shape)
            idx=np.argmax(output[-1])
            seq.append(idx)
            print seq
            print [ d2w[s] for s in seq ]
            if idx==2:
                break

if __name__ == '__main__':
    print sys.argv[1]
    if sys.argv[1] == 'train':
        train()
    else:
        predict()
