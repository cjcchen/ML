from datetime import datetime
import time
from matplotlib.pyplot import *

import matplotlib.pyplot as plt
import tensorflow as tf
import resnet 

from image_tools import *
from data_tools import *
from voc_input import *


def predict_signle(sessioin, model, x,y):
    predict= session.run([model.predict], feed_dict={model.x:x,model.y:y,resnet.is_training:False})
    class_type = np.argmax(predict[0],axis=1)
    acc_count=np.sum(class_type == y)
    acc=acc_count/float(len(y))*100 

    return acc_count, acc

def get_train_op(loss,global_step):
    lrn_rate = 0.01

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(loss, trainable_variables)


    optimizer = tf.train.MomentumOptimizer(lrn_rate, 0.9)
    grads = optimizer.compute_gradients(loss)

    for grad, var in grads:
        if grad is not None :
            tf.summary.histogram(var.op.name + '/gradients', grad)
            print grad
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    train_ops = [apply_gradient_op] + resnet.extra_train_ops
    return tf.group(*train_ops)

def load_session(checkpoint_dir):
    session = tf.Session()

    try:
        print("Trying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        saver.restore(session, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())

    return session

def train(model, data_x, data_y,test_x,test_y, class_names):

    train_len = len(data_x)/5*4
    train_x = data_x[0:train_len]
    train_y = data_y[0:train_len]
    valid_x = data_x[train_len:]
    valid_y = data_y[train_len:]

    save_dir = 'checkpoints/224l_64b_5n/'
    save_path = save_dir+"resnet"
    log_dir='logs/224_res_64b_5n'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    merged_summary_op = tf.summary.merge_all()

    loss = model.loss;

    train_op = get_train_op(loss,global_step)

    session = load_session(save_dir)

    saver = tf.train.Saver()
    summary = tf.Summary()
    summary_writer = tf.summary.FileWriter(log_dir, session.graph)


    print train_x.shape,valid_x.shape

    while True:
        x,y, = random_batch(train_x,train_y,batch_size)

        v_x,v_y = random_batch(valid_x,valid_y,batch_size)

        feed_dict_train = {model.x: x, model.y:y}
        disorder_img = session.run([model.disorder_img], feed_dict=feed_dict_train)
        disorder_img = np.array(disorder_img[0])
        print disorder_img.shape, y.shape
        feed_dict_train = {model.x: disorder_img, model.y:y, resnet.is_training:True}
        i_global, _,summary_str, c_loss = session.run([global_step, train_op, merged_summary_op,loss], feed_dict=feed_dict_train)
        if(i_global%10==0):
            predict_acc_count, predict_acc = predict_signle(session,model, x,y)
            v_predict_acc_count, v_predict_acc = predict_signle(session,model, v_x,v_y)
            print i_global, predict_acc, v_predict_acc

            summary.value.add(tag='acc', simple_value=pre_dict_acc)
            summary.value.add(tag='val acc', simple_value=v_predict_acc)
        summary_writer.add_summary(summary_str, i_global)
        summary_writer.add_summary(summary, i_global)
        if i_global%50 == 0:
            saver.save(session, save_path=save_path, global_step=i_global) 
            print("Saved checkpoint.")


    summary_writer.close()

def predict(model, test_x,test_y, class_names):

    print test_x.shape,test_y.shape
    test_x=test_x[0:256]
    test_y=test_y[0:256]
    
    save_dir = 'checkpoints/small_64b_5n/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss = model.loss;
    session = tf.Session()
    saver = tf.train.Saver()

    try:
        print("Trying to restore last checkpoint ...")
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



def main(argv=None):  
    x,y,l=get_voc_data()
    #test_x,test_y,test_l=get_data_set("test",100)
    print "x shape:",x.shape
    print "y shape:",y.shape
    
    #plot_img(x,y,l)

    model = resnet.SamllResNet()

    #predict(model, test_x,test_y, test_l)
    train(model,x,y,test_x,test_y,l)



if __name__ == '__main__':
    tf.app.run()
