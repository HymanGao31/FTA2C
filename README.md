#  FTA2C:Achieving Superior Trade-off between Accuracy and Robustness in Adversarial Training
## Updates
[05/2023] Code is released.

## Setup
The packages necessary for running our code are provided in `environment.yml`. Create the conda environment `FTA2C` by running:
```
conda env create -f environment.yml
```
Note that if you're using more latest GPUs (e.g., RTX 3090), you may need to refer to [this pytorch link](https://pytorch.org/get-started/locally/) to install the PyTorch package that suits your cuda version.

## Running FTA2C
All codes for training and testing our FTA2C are provided in `run.sh`.

In this code, we only support running FTA2C with a single GPU, as our FTA2C module is light enough to be run on one GPU. 
While not implemented and tested, I expect that our code can also be run on multiple GPUs using `torch.nn.DataParallel()`.

### Training
The codes for training our FTA2C module can be found in `train.py`. 
Below is one example of training FTA2C on ResNet-18 using CIFAR-10 dataset:
```
python train.py \
--save_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
```
Please refer to `run.sh` for more training scripts and to `train.py` for more detailed arguments and their descriptions.

### Testing
We load this checkpoint to evaluate the robustness of FTA2C module.
Below is one example of testing FTA2C on ResNet-18 using CIFAR-10 dataset:
for Ensemble Attack:
```
python test.py \
--load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
```
for FGSM/PGD/CW Attack:
```
python pgd_attack_fta2c.py \
--model resnet18 --dataset cifar10 --white_box_attack True --load_name cifar10_resnet18_best
```
Please refer to `run.sh` for more testing scripts and to `test.py` for more detailed arguments and their descriptions.

## Citation
If you find our work useful, please consider using the following citation:


## Acknowledgement
We thank the authors of [CAS](https://github.com/bymavis/CAS_ICLR2021) and [FSR](https://github.com/wkim97/FSR) for their contribution to the field and our research. 
Our implementation is inspired by or utilized parts of CAS and FSR as credited in our code.
