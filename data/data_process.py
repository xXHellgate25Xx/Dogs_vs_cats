import os
import cv2
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# dir_train = "data/train"
# dir_test = "data/test1"

# imgs = os.listdir(dir_train)
# test_imgs = os.listdir(dir_test)

#Augmentations
def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize((0,0,0),(1,1,1)) #-> Standardization as mean = [0,0,0] and std = [1,1,1] for each channel R,G,B
    ])

def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0,0,0),(1,1,1))
    ])
#

class DogCatDataset(Dataset):
    def __init__(self, imgs, class_to_int, dir_train, mode="train", transforms= None):
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        self.dir_train = dir_train
    
    def __getitem__(self,idx): #For this example: a[idx]
        image_name = self.imgs[idx]
        
        ### Reading, converting and normalizing image for OpenCV
        #img = cv2.imread(DIR_TRAIN + image_name, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (224,224))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        #img /= 255.

        img = Image.open(self.dir_train + "/" + image_name)
        img = img.resize((224,224))
        if self.mode == "train" or self.mode == "val":
            label = self.class_to_int[image_name.split(".")[0]]
            
            #print(label)
            #print(type(label))
            
            label = torch.tensor(label, dtype= torch.float32)
            img = self.transforms(img)
            return img,label
        else:
            img = self.transforms(img)
            return img

    def __len__(self):
        return len(self.imgs)

