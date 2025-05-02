python train.py --save_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
python pgd_attack_fta2c.py --model resnet18 --dataset cifar10 --white_box_attack True --load_name cifar10_resnet18_best

python train.py --save_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 0
python test.py --load_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 0
python pgd_attack_fta2c.py --model vgg16 --dataset cifar10 --white_box_attack True --load_name cifar10_vgg16_best--

python train.py --save_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 0 --is_train=True
python test.py --load_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 0
python pgd_attack_fta2c.py --model wideresnet34 --dataset cifar10 --white_box_attack True --load_name cifar10_wideresnet34_best

python train.py --save_name svhn_resnet18 --dataset svhn --model resnet18 --device 0 --lr 0.01 --alpha 0.125
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 0
python pgd_attack_fta2c.py --model resnet18 --dataset svhn --white_box_attack True --load_name svhn_resnet18_best

python train.py --save_name svhn_vgg16 --dataset svhn --model vgg16 --device 0 --lr 0.01 --alpha 0.125
python test.py --load_name svhn_vgg16 --dataset svhn --model vgg16 --device 0

python train.py --model wideresnet34 --dataset svhn --save_name svhn_wideresnet34 --device 0 --lr 0.01 --alpha 0.125
python test.py --load_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 0
python pgd_attack_fta2c.py --model wideresnet34 --dataset svhn --white_box_attack True --load_name svhn_wideresnet34_best

python train_trades_fta2c.py  --dataset ImageNet  --model resnet18 --device 0  --batch_size 16 --lr 0.01

python train_trades_fta2c.py  --dataset cifar10 --model resnet18/vgg --device 0
python pgd_attack_fta2c.py --model resnet18/vgg --dataset cifar10 --white_box_attack True

python train_trades_fta2c.py --dataset cifar10 --model wideresnet --device 0 --lr 0.05
python train_mart_fta2c.py --dataset cifar10 --model wideresnet --device 0 --lr 0.01

python train_trades_fta2c.py  --dataset svhn --model resnet18/vgg/wideresnet --device 0  --lr 0.01 --step_size 0.004
python pgd_attack_fta2c.py --model resnet18/vgg/wideresnet --dataset svhn --white_box_attack True

python train_mart_fta2c.py  --dataset cifar10 --model resnet18/vgg --device 0
python pgd_attack_fta2c.py --model resnet18/vgg --dataset cifar10 --white_box_attack True
 
python train_mart_fta2c.py  --dataset svhn --model resnet18/vgg/wideresnet --device 0  --lr 0.01 --step_size 0.004
python pgd_attack_fta2c.py --model resnet18/vgg/wideresnet --dataset svhn --white_box_attack True

python train_trades_fta2c.py --dataset ImageNet --model resnet18 --device 0 --batch_size 16 --lr 0.01
python train_mart_fta2c.py  --dataset ImageNet --model resnet18 --device 0 --batch_size 16  --lr 0.01


