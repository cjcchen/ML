#python train_zfnet.py --data_dir="/media/junechen/TOSHIBA EXT/ILSVRC2012/" --meta_data_dir=../dataset/data/imagenet/meta/ --test_data_dir=../dataset/data/imagenet/val --batch_size 64 --save_dir checkpoints/zfnet3/ --log_dir logs/zfnet3/ --lr=0.01 --cpu_mode=gpu
python train_zfnet.py --data_dir="./data/" --meta_data_dir=../dataset/data/imagenet/meta/ --test_data_dir=../dataset/data/imagenet/val --batch_size 64 --save_dir checkpoints/zfnet/ --log_dir logs/zfnet/ --lr=0.001 --cpu_mode=gpu
