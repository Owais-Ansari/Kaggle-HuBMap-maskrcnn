import os
import numpy as np
import json
import torch
import glob
import json
from PIL import Image
from torch.utils.data import Dataset
from skimage.measure import label
from skimage.draw import polygon
import pandas as pd



class ImageHuBMAPDataset(Dataset):
    def __init__(self, image_dir, labels_file, tfms =  None, split = None, ignore_label=255,config = None):
        
        self.image_dir   = image_dir
        self.labels_file = labels_file
        self.tfms  = tfms
        self.split = split
        self.ignore_label = ignore_label
        

        self.df = pd.read_csv(config.root_dir + 'tile_meta.csv')
        if self.split == 'train':
            self.df = list(self.df[self.df['dataset']==1]['id']) #+ list(self.df[self.df['dataset']==2]['id'])
        elif self.split == 'val':
            self.df = list(self.df[self.df['dataset']==2]['id']) #+ list(self.df[self.df['dataset']==2]['id'])
            
            
        with open(labels_file, 'r') as json_file:
            self.json_labels = [json.loads(line) for line in json_file if json.loads(line)['id'] in self.df]
            
            
    def __len__(self):
        return len(self.json_labels)

    def __getitem__(self, idx):

        img = np.asarray(Image.open(self.image_dir + self.json_labels[idx]['id'] + '.tif'), dtype = 'uint8')
        img = img.copy() 

        mask = np.zeros([512,512],dtype  = 'uint8')
        
        
        for annot in self.json_labels[idx]['annotations']:
            
            cords = annot['coordinates']
            if annot['type'] == "blood_vessel":
                for inst, cd in enumerate(cords):
                    rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
                    rr, cc = polygon(rr, cc)
                    #masks_type[0,rr, cc ] = 1
                    mask[rr, cc] = 1
 
            elif annot['type'] == 'glomerulus':
                for inst, cd in enumerate(cords):
                    rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
                    rr, cc = polygon(rr, cc)
                    #masks_type[1,rr, cc] = 2
                    mask[rr, cc] = 2
            elif annot['type'] == 'unsure':
                for inst, cd in enumerate(cords):
                    rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
                    rr, cc = polygon(rr, cc)
                    #masks_type[2,rr, cc] = self.ignore_label
                    mask[rr, cc] = self.ignore_label

        
        # ignoring unsure region
        #img[mask==self.ignore_label,:] = [235,221,225] 
        #mask[mask==self.ignore_label] = 0 
                           
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img, mask  = augmented['image'],augmented['mask']
            
           
        
        instances = label(mask, background=0)
        #excluding label 0
        objects =  np.unique(instances)[1:]
        num_objects =  len(objects)
        masks_all = np.zeros([num_objects,512,512], dtype = 'uint8')
        labels    =  np.zeros([num_objects], dtype = 'uint8')
        
        for instance in objects:
            masks_all[instance-1, instances == instance]=1
            instance_labels  =  (instances==instance).astype('uint8')#*mask
            instance_label   = int(np.unique(instance_labels)[1])
            #labels.append(instance_label)
            labels[instance-1] = instance_label
            
        #get bounding box coordinates for each mask

        boxes = []
        
        for i in range(num_objects):
            pos  = np.where(masks_all[i,:,:]>0)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        #convert everything into a torch.Tensor
        
        boxes  = torch.as_tensor(boxes,     dtype=torch.float32)
        labels = torch.as_tensor(labels,    dtype=torch.int64)
        masks  = torch.as_tensor(masks_all, dtype=torch.uint8)

        image_id = torch.tensor([idx])
                                          
        area = (boxes[:,3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {}
        target["boxes"]    = boxes
        target["labels"]   = labels
        target["masks"]    = masks
        target["image_id"] = image_id
        target["area"]     = area
        target["iscrowd"]  = iscrowd
        
        return img, target



class get_test_dataset(Dataset):
    def __init__(self, image_path,tfms = None):                       
                       
        self.imgs_path = [img_p for img_p in glob.glob(image_path + '/*.tif')]

        self.tfms = tfms
        self.image_path = image_path
        
        
    def __len__(self):
        return len(self.imgs_path)
                         
    def __getitem__(self, idx):
        
        img_path = self.imgs_path[idx]
        img_name = os.path.basename(img_path)
        img  = np.asarray(Image.open(img_path),dtype=np.uint8)
        if self.tfms is not None:
            #augmented = self.tfms()
            augmented = self.tfms(image=img)
            #img,mask = augmented(img, mask)
            img = augmented['image']
        #return {'image':img, 'path':self.image_path}  
        
        return img, img_path



