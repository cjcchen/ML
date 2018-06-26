from baseline_solver import CaptioningSolver
from baseline import CaptionGenerator
import tensorflow as tf
import data

from vgg19 import Vgg19

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def main():
    train_caption_path = "/home/tusimple/junechen/ml_data/data/annotations/captions_train2014.json"
    train_image_path="/home/tusimple/junechen/ml_data/data/train2014/"

    f, image,label,word, target, w2d, d2w= data.get_data(train_caption_path, train_image_path, max_len=16+1, batch_size=50, mode='train')

    cnn_net = Vgg19()
    cnn_net.build(image)
    image_feature = cnn_net.conv5_3 #[-1,14*14,512]

    print ("cnn build done")
    model = CaptionGenerator(image_feature, label, w2d, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    print ("build done")
    solver = CaptioningSolver(model, data, None, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.0005, print_every=1000, save_every=1, image_path='./image/')

    solver.train()

if __name__ == "__main__":
    main()
