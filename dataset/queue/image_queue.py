import tensorflow as tf

def image_queue(image, label, batch_size, threads_num = 3, mode='train'):
   if mode == 'train':
       queue = tf.RandomShuffleQueue(
       capacity= (threads_num +3)* batch_size,
               min_after_dequeue= batch_size,
               dtypes=[tf.float32,tf.int32],
               shapes=[image.shape, label.shape])
   else:
       threads_num = 1
       queue = tf.FIFOQueue(
               (threads_num + 3) * batch_size,
               dtypes=[tf.float32, tf.int32],
                shapes=[image.shape, label.shape])

   enqueue_op = queue.enqueue([image, label])
   tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, [enqueue_op] * threads_num))
   image, label = queue.dequeue_many(batch_size)
   return  image, label

def data_queue(data, dtype, shape, batch_size, threads_num = 3, mode='train'):
   if mode == 'train':
       queue = tf.RandomShuffleQueue(
       capacity= (threads_num +3)* batch_size,
               min_after_dequeue= batch_size,
               dtypes=dtype,
               shapes=shape)
   else:
       threads_num = 1
       queue = tf.FIFOQueue(
               (threads_num + 3) * batch_size,
               dtypes=dtype,
                shapes=shape)

   enqueue_op = queue.enqueue(data)
   tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, [enqueue_op] * threads_num))
   return queue.dequeue_many(batch_size)

