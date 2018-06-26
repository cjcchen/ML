from solver import CaptioningSolver
from base_model import CaptionGenerator
from core.utils import load_coco_data

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class img_config():
    hidden_size=1024
    embedding_size=512
    feature_dim = [196,512]
    seq_len = 16


path="/home/tusimple/junechen/ml_data/data/show"
def main():
    config = img_config()

    # load train dataset
    data = load_coco_data(data_path=path, split='train')
    print "load coco data done"
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path=path, split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)


    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=100, save_every=1, image_path='%s/'%path,
                                    pretrained_model=None, model_path='%s/model5/lstm/'%path, test_model='%s/model5/lstm/model-10'%path,
                                     print_bleu=True, log_path='%s/log5/'%path)

    solver.train()

if __name__ == "__main__":
    main()
