
import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2



train_aug = A.Compose([
    #A.Resize(1024, 1024),
    A.Equalize(p=0.05),
    A.HueSaturationValue(),
    A.ColorJitter(),
    A.RandomBrightnessContrast(),
    A.ChannelShuffle(),
    #A.RandomCrop(width=256, height=256),
    A.Flip(),
    A.RandomRotate90(),
    #A.Transpose(),
    #Rotate(border_mode = cv2.BORDER_REFLECT),
    A.GaussNoise(),
    #RGBShift(),
    A.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])

valid_aug = A.Compose([
    #Resize(512, 512),
    A.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])














