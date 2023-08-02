import torch
import numpy as np
from inference import Config
from torchvision.models.detection import MaskRCNN
from dataloader import get_test_dataset
from torch.utils.data import DataLoader
from utils import get_file_dir, convnext_fpn_backbone
from utilities.augment import train_aug, valid_aug
import os
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from pycocotools import _mask as coco_mask
import zlib, base64
import pandas as pd



def encode_binary_mask(mask):
    if mask.dtype != bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str



backbone = convnext_fpn_backbone(
    Config.backbone,
    Config.trainable_layers)


model = MaskRCNN(
    backbone, 
    num_classes=Config.num_classes, 
    max_size=Config.max_size,
    min_size=Config.min_size,)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
model_weights = torch.load(Config.weights)['model_state_dict']
model.load_state_dict(model_weights,strict = True);
model.to(Config.device)



validation_dataset = get_test_dataset(Config.test_dir, tfms = valid_aug)
valloader = DataLoader(validation_dataset, batch_size=Config.train_bs, 
                          shuffle=True, pin_memory=True, 
                          drop_last=False, collate_fn=lambda x: tuple(zip(*x)))

confidence = 0.3
mask_th=0.60
ids = []
heights = []
widths = []
pred_strings = []
model.eval()
for imgs,paths in valloader:
    inputs = list(image.to(Config.device) for image in imgs)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
    for id, output in enumerate(outputs):
        pred = output        
        pred_score = list(pred['scores'].detach().cpu().numpy())
        indices = [pred_score.index(x) for x in pred_score if x>confidence][-1]
        masks = (pred['masks']>mask_th).squeeze().detach().cpu().numpy()
        pred_class = [Config.class_names[i] for i in list(pred['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().cpu().numpy())]
        masks      = masks[:indices+1]
        pred_boxes = pred_boxes[:indices+1]
        pred_class = pred_class[:indices+1]
        pred_score = pred_score[:indices+1]
        pred_string = ''
        _,w,h = masks.shape
        filename = os.path.basename(paths[id]).replace('.tif','')
        masks_stacked = np.zeros([512,512],dtype = 'uint8')
        for indx in range(masks.shape[0]):
            if pred_class[indx]=='blood_vessels':
                encode = encode_binary_mask(masks[indx,:,:])
                if indx == 0:
                    pred_string += f"0 {pred_score[indx]} {encode.decode('utf-8')}"
                else:
                    pred_string += f" 0 {pred_score[indx]} {encode.decode('utf-8')}"
                masks_stacked[masks[indx,:,:]==1]=255                    
        ids.append(filename)
        pred_strings.append(pred_string)
        heights.append(h)
        widths.append(w)
        
      
submission = pd.DataFrame({'id':ids,
                           'height': heights,
                           'width': widths,
                           'prediction string':pred_strings}
                           )
        
submission = submission.set_index('id')
submission.to_csv('/home/owaishs/temp/submission.csv') 
#Image.fromarray(masks_stacked).save('/home/owaishs/temp/'+ filename + '.png')  

    
    
    
    