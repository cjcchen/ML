import rpn
import os
import tensorflow as tf
from dataset.voc import voc_input
from zfnet import zfnet

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cpu_mode', 'cpu', "cpu? or gpu?")
tf.app.flags.DEFINE_string('data_dir', '', "data dir")
tf.app.flags.DEFINE_string('pre_model_dir', '../zfnet/checkpoints/zfnet/', "data dir")
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/small_64b_5n/', "check point dir")
tf.app.flags.DEFINE_string('log_dir', 'logs/cifar_128_5n/', "check point dir")
tf.app.flags.DEFINE_float('lr', 0.1, "learning rate")

file_name, image, gt, img_info = voc_input.get_voc_data(FLAGS.data_dir, mode = 'train')
test_file_name, test_image, test_gt, test_img_info = voc_input.get_voc_data(FLAGS.data_dir, mode = 'val')

def data_func(session, model):
    [f, train_image, train_gt_box, train_img_size] = session.run([file_name, image,gt,img_info])
    print "train data:",f, train_image.shape, train_gt_box.shape, train_img_size.shape
    
    pre_sess = model.pre_model.get_session() 
    y = model.pre_model.get_conv_layer(-1)

    feed_in = {model.pre_model.x: train_image}
    [output] = pre_sess.run([y], feed_dict=feed_in)

    return {model.images:output, model.gt_box:train_gt_box, model.img_size:train_img_size}

def main(argv=None):
    with tf.device("/"+FLAGS.cpu_mode+":0"):
        images=tf.placeholder(tf.float32, [1,None,None,3])
        with tf.variable_scope("zfnet", reuse=None):
            pre_model = zfnet.ZFNet(images,None, 0, "tunning", "zfnet")

        feature_map=tf.placeholder(tf.float32, [1,None,None,256])
        gt_box = tf.placeholder(tf.float32,[1,None,5])
        img_size = tf.placeholder(tf.float32,[None,3])
        with tf.variable_scope("rpn", reuse=None):
            model = rpn.RPN (pre_model, feature_map, gt_box, img_size,"rpn")

        session = tf.Session()
        
        pre_model.set_session(session)
        assert pre_model.load_model(FLAGS.pre_model_dir) == 0

        model.set_session(session)
        model.set_para(FLAGS.save_dir, FLAGS.log_dir, FLAGS.lr)

        model.train(data_func)

if __name__ == '__main__':
    tf.app.run()
