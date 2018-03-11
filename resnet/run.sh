#python train_cifar.py --data_dir=../dataset/data/cifar_10_bin/ --batch_size 128 --save_dir checkpoints/cifar10_128_5n_test/ --log_dir logs/cifar10_128_5n_test/ --lr=0.1
python train_imagenet.py --data_dir=../dataset/data/ILSVRC2012_devkit_t12/data/ --batch_size 128 --save_dir checkpoints/cifar10_128_5n_test/ --log_dir logs/cifar10_128_5n_test/ --lr=0.1
