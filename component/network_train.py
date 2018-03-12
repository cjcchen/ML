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
    predict= session.run([model.predict], feed_dict={model.x:x,model.y:y})
    class_type = np.argmax(predict[0],axis=1)
    acc_count=np.sum(class_type == y)
    acc=acc_count/float(len(y))*100 

    return acc_count, acc

def get_collection():

    for key in ['stddev','max','min','mean']:
        value = tf.get_collection(key)
        for v in value:
          tf.summary.scalar(v.op.name, v)

    for key in ['histogram']:
        value = tf.get_collection(key)
        for v in value:
            tf.summary.histogram(v.op.name, v)

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
    summary_train = tf.Summary()
    summary_eval = tf.Summary()

    summary_writer = tf.summary.FileWriter(log_dir, session.graph)
    summary_writer_train = tf.summary.FileWriter(log_dir+"/train", session.graph)
    summary_writer_eval = tf.summary.FileWriter(log_dir+"/eval", session.graph)


    while True:
        x,y = session.run([data_x,data_y])
        feed_dict_train = {model.x: x, model.y:y}
        i_global, _,summary_str, c_loss = session.run([global_step, train_op, merged_summary_op,model.loss], feed_dict=feed_dict_train)
        summary_writer.add_summary(summary_str, i_global)
        if i_global%100 == 0:
            t_x,t_y = session.run([test_x,test_y])

            acc_count, acc = eval_predict(session, model, x,y)
            t_acc_count, t_acc = eval_predict(session, eval_model, t_x,t_y)

            summary.value.add(tag='acc', simple_value=acc)
            summary.value.add(tag='eval acc', simple_value=t_acc)
            #summary.value.add(tag='test acc', simple_value=precision)

            #summary_train.value.add(tag='acc', simple_value=acc) #add to train summary
            summary_eval.value.add(tag='acc', simple_value=t_acc)#add to eval summary

            summary_writer.add_summary(summary, i_global)

            #summary_writer_train.add_summary(summary_train, i_global)#write train to tensorboard
            summary_writer_eval.add_summary(summary_eval, i_global)#write eval to tensorboard

            print "step:",i_global,"lr:",FLAGS.lr,"loss:",c_loss, "test acc:",t_acc, "train acc:",acc
            saver.save(session, save_path=save_path, global_step=i_global) 
            print("Saved checkpoint.")

    summary_writer.close()

def eval():

    test_x,test_y=cifar_input.image_input(FLAGS.data_dir, batch_size=FLAGS.batch_size,data_type='test')

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

    for op in tf.get_default_graph().get_operations():
      print str(op.name)


    saver = tf.train.Saver()
    session = load_session(save_dir, saver)
    threads = tf.train.start_queue_runners(session)

    eval_account = 0
    correct_account = 0
    while eval_account<10000:
            t_x,t_y = session.run([test_x,test_y])

            t_acc_count, t_acc = eval_predict(session, eval_model, t_x,t_y)

            correct_account += t_acc_count
            eval_account += len(t_x)

            print "eval account:", eval_account,"test acc:",t_acc, "total:", float(correct_account)/eval_account*100




