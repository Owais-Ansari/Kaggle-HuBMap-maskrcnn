<br />
<p align="center">
  
  <h1 align="center">  Kaggle-HuBMap: Maskrcnn with ConvNext backbone</h1>

  <p align="center">
    Segment intstances of microvascular structures from healthy human kiney tissue slides.
    <br />
  </p>
</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


<!-- GETTING STARTED -->
## Getting Started

Follow these simple example steps to get a local copy up and to run.


### Installation

1. Clone the repo
```sh
git clone https://github.com/Owais-Ansari/Kaggle-HuBMap-maskrcnn.git
```
2. Check if you have a virtual env 
```sh
virtualenv --version
```
3. If (not Installed) 
```sh
pip install virtualenv
```
4. Now create a virtual env in cd Kaggle-HuBMap-maskrcnn/
```sh
virtualenv venv
```
5. Then download a python modules
```sh
pip install -r requirements.txt
```

# Usage

## Training


Update the Inference.py


```
import torch

class Config:
    """
    MaskRCNN model configuration class. 
    Enter specific configurations for the own model.
    """
    seed              = 101
    model_name        = 'MaskRCNN'
    backbone          = 'convnext_tiny'
    save_path         = '../HuBMAP/hubmap-hacking-the-human-vasculature/checkpoints/'
    class_names       = ['bkg', 'blood_vessels', 'glomerulus']
    weights           = '' # file path with weights of your pre-trained model
    pretrained        = False # If you have a pre-trained model, load weights
    max_size          = 512
    min_size          = 512
    trainable_layers  = 8 # [0,8]
    train_bs          = 4# train batch size
    val_bs            = 4 # validation batch size
    split_size        = 0.2 # If you do not have a validation dataset, enter the split size between 0 and 1
    epochs            = 50
    lr                = 5e-4
    momentum          = 0.9
    min_lr            = 1e-8
    step_size         = 3
    gamma             = 0.1
    T_max             = int(100*6*1.8)
    T_0               = 25
    weight_decay      = 5e-6
    n_accumulate      = 32//train_bs
    n_fold            = 5
    num_classes       = 3
    device            = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    verbose_num       = 10
    root_dir = '../HuBMAP/hubmap-hacking-the-human-vasculature/'
    test_dir = '../HuBMAP/hubmap-hacking-the-human-vasculature/test/'
```
```
python train.py
```

## Prediction

```
import torch

class Config:
    """
    MaskRCNN model configuration class. 
    Enter specific configurations for the own model.
    """
    seed              = 101
    backbone          = 'convnext_tiny'
    class_names       = ['bkg', 'blood_vessels', 'glomerulus']
    weights           = '../HuBMAP/hubmap-hacking-the-human-vasculature/checkpoints/MaskRCNN.pth' # file path with weights of your pre-trained model
    pretrained        = False # If you have a pre-trained model, load weights
    max_size          = 512
    min_size          = 512
    val_bs            = 4 # validation batch size
    num_classes       = 3
    device            = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    verbose_num       = 10
    test_dir = '../HuBMAP/hubmap-hacking-the-human-vasculature/test/'
```

```
python predict.py
```


