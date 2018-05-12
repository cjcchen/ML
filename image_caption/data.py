import tensorflow as tf
import json
import os

def read_caption_data(caption_file):
    with open(caption_file) as f:
        caption_data = json.load(f)
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    image_files=[]
    captions=[]
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        image_files.append(id_to_filename[image_id])
        captions.append(annotation['caption'])
    image_files=image_files[0:10]
    captions=captions[0:10]
    print ("len",len(image_files), len(captions))
    file_path, label = tf.train.slice_input_producer([image_files,captions], shuffle=True)
    images = tf.image.decode_jpeg(tf.read_file(file_path), channels=3)
    return images, captions

def process(images, captions):
    return images, captions

def get_data(caption_file, mode = 'train', threads_num=1, batch_size=1):
    images, captions = read_caption_data(caption_file)
    images, labels = process(images, captions)
    print (images.shape)
    #print (labels)

    if mode == 'train':
       queue = tf.RandomShuffleQueue(
       capacity= (threads_num +3)* batch_size,
               min_after_dequeue= batch_size,
               dtypes=[tf.float32,tf.int32],
               shapes=[images.shape, []])
    else:
       threads_num = 1
       queue = tf.FIFOQueue(
               (threads_num + 3) * batch_size,
               dtypes=[tf.float32, tf.int32],
                shapes=[images.shape, labels.shape])

    enqueue_op = queue.enqueue([images, labels])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, [enqueue_op] * threads_num))
    images, labels = queue.dequeue_many(batch_size)
    return  images, labels



if __name__ == '__main__':
    image,label = get_data("/home/junechen/ml/annotations/captions_train2014.json")

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess)
        d,l = sess.run([data,label])
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        imgplot = plt.imshow(d)
        plt.show() 
#_process_caption_data("/home/junechen/ml/annotations/captions_train2014.json", "/home/junechen/ml/",100)
