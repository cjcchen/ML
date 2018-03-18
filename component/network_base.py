from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import os
import component

class BaseNet:
    def __init__(self,pre_fix):
        self.pre_fix=pre_fix
        self.train_ops = []
    
    def set_para(self,model_path, log_path, lr):
        self.model_path = model_path
        self.log_dir = log_path
        self.lr = lr
        
        self.save_dir = self.model_path 
        self.save_path = self.save_dir+"model"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_model(self, model_path):
        return self.load_session(model_path)

    def get_session(self):
        return self.session

    def set_session(self,session):
        self.session = session

    def load_session(self,checkpoint_dir):
        self.saver = tf.train.Saver([v for v in tf.all_variables() if v.name.startswith(self.pre_fix+'/') ])
        try:
            print("Trying to restore last checkpoint ...:",checkpoint_dir)
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            self.saver.restore(self.session, save_path=last_chk_path)
            print("restore last checkpoint %s done"%checkpoint_dir)
            return 0
        except Exception as e:
            print("Failed to restore checkpoint. Initializing variables instead."),e
            self.session.run(tf.global_variables_initializer())
            return 1

    def get_train_op(self,global_step, lr_rate, moment_rate=0.9):
        optimizer = tf.train.MomentumOptimizer(lr_rate,moment_rate)
        grads = optimizer.compute_gradients(self.loss)

        for grad, var in grads:
            if grad is not None :
                tf.summary.histogram(var.op.name + '/gradients', grad)
                print grad
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        train_ops = [apply_gradient_op] + self.train_ops
        return tf.group(*train_ops)

    def train(self, data_func, eval_func = None):
        summary_op = component.get_collection_with_prefix(self.pre_fix)

        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        merged_summary_op = tf.summary.merge(summary_op)

        train_op = self.get_train_op(global_step, self.lr)

        self.load_session(self.model_path)

        session = self.session
        threads = tf.train.start_queue_runners(session)

        summary = tf.Summary()
        summary_eval = tf.Summary()

        summary_writer = tf.summary.FileWriter(self.log_dir, session.graph)
        summary_writer_eval = tf.summary.FileWriter(self.log_dir+"/eval", session.graph)

        while True:
            try:
                feed_in = data_func(session, self)
                i_global, _,summary_str, c_loss = session.run([global_step, train_op, merged_summary_op,self.loss], feed_dict=feed_in)
                print "step:",i_global, "loss:",c_loss
                summary_writer.add_summary(summary_str, i_global)

                if i_global%10 == 0:
                    if eval_func is not None:
                        train_acc, eval_acc = eval_func(session, self)

                        summary.value.add(tag='acc', simple_value=acc)

                        summary_eval.value.add(tag='acc', simple_value=t_acc)#add to eval summary
                        summary_writer_eval.add_summary(summary_eval, i_global)#write eval to tensorboard

                    summary.value.add(tag='cost', simple_value=c_loss)
                    summary_writer.add_summary(summary, i_global)

                    self.saver.save(session, save_path=self.save_path, global_step=i_global) 
                    print("Saved checkpoint.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print "run fail, continue",e
                print feed_in
                continue

        summary_writer.close()

