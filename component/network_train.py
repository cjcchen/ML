from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/small_64b_5n/', "check point dir")
tf.app.flags.DEFINE_string('log_dir', 'logs/cifar_128_5n/', "check point dir")
tf.app.flags.DEFINE_float('lr', 0.1, "learning rate")


def eval_predict(session, model, x,y):
    predict= session.run([model.predict], feed_dict={model.x:x,model.y:y,resnet.is_training:False})
    class_type = np.argmax(predict[0],axis=1)
    acc_count=np.sum(class_type == y)
    acc=acc_count/float(len(y))*100 

    return acc_count, acc

def get_collection():

    for key in ['stddev','max','min','mean']:
        value = tf.get_collection(key)
        print "get collect:",value	
        for v in value:
            tf.summary.scalar(key, v)

    for key in ['histogram']:
        value = tf.get_collection(key)
        for v in value:
            print key,v
            tf.summary.histogram(key, v)

def load_session(checkpoint_dir, saver):
    session = tf.Session()

    try:
        print("Trying to restore last checkpoint ...:",checkpoint_dir)
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        saver.restore(session, save_path=last_chk_path)
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())

    return session

def train(model, eval_model,data_x, data_y, test_x, test_y):


    save_dir = FLAGS.save_dir
    save_path = save_dir+"resnet"
    log_dir=FLAGS.log_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    get_collection()

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    merged_summary_op = tf.summary.merge_all()

    train_op = model.get_train_op(global_step, FLAGS.lr)

    saver = tf.train.Saver()
    session = load_session(save_dir, saver)
    threads = tf.train.start_queue_runners(session)

    summary = tf.Summary()
    summary_writer = tf.summary.FileWriter(log_dir, session.graph)


    while True:
        x,y = session.run([data_x,data_y])
        print len(x), len(y)
        feed_dict_train = {model.x: x, model.y:y}
        i_global, _,summary_str, c_loss = session.run([global_step, train_op, merged_summary_op,model.loss], feed_dict=feed_dict_train)
        #i_global, _,summary_str, c_loss, precision = session.run([global_step, train_op, merged_summary_op,model.loss, model.precision])
        summary_writer.add_summary(summary_str, i_global)
        print "acc:",precision
        if i_global%10 == 0:
            t_x,t_y = session.run([test_x,test_y])
            #t_acc_count, t_acc = eval_predict(session, test_model, x,y)
            acc_count, acc = eval_predict(session, test_model, t_x,t_y)

            summary.value.add(tag='acc', simple_value=t_acc)
            summary.value.add(tag='test acc', simple_value=acc)
            summary_writer.add_summary(summary, i_global)

            acc = precision

            print "step:",i_global,"lr:",FLAGS.lr,"loss:",c_loss, "test acc:",acc, "train acc:",t_acc
            saver.save(session, save_path=save_path, global_step=i_global) 
            print("Saved checkpoint.")

    summary_writer.close()

def predict(model, test_x,test_y, class_names):

    print test_x.shape,test_y.shape
    test_x=test_x[0:256]
    test_y=test_y[0:256]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss = model.loss;
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    saver = tf.train.Saver()

    try:
        print("Trying to restore last checkpoint ...",save_dir)
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        saver.restore(session, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())


    while True:
        test_predict= session.run([model.predict], feed_dict={model.x:test_x,model.y:test_y,resnet.is_training:False})
        test_class_type = np.argmax(test_predict[0],axis=1)
        print np.sum(test_class_type==test_y)/float(len(test_y))*100
        break
