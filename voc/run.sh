#python train_imgnet.py --data_dir=data/cifar_10_bin/ --batch_size 128 --save_dir checkpoints/cifar10_128_test8/ --log_dir test_log/cifar10_128_50/ --lr=0.001 --cpu_mode=gpu
python train_imgnet.py --data_dir=data/cifar_10_bin/ --batch_size 100 --save_dir checkpoints/cifar10_128_test8/ --log_dir test_log/cifar10_128_50/ --cpu_mode=gpu --run_mode=eval
