from data.data_process import *
from model.model import *
from utils.utils import *
from utils.train_utils import *

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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', default=2, type=int)
args = parser.parse_args()

#Prepare data

dir_train = "data/train"
dir_test = "data/test1"

imgs = os.listdir(dir_train)
test_imgs = os.listdir(dir_test)

#print(imgs[:5])
#print(test_imgs[:5])

train_imgs, val_imgs = train_test_split(imgs, test_size = 0.25)

#print(train_imgs[:5])
#print(val_imgs[:5])

train_dataset = DogCatDataset(train_imgs, class_to_int, dir_train, mode="train", transforms=get_train_transform())
val_dataset = DogCatDataset(val_imgs, class_to_int, dir_train, mode="val", transforms=get_train_transform())
test_dataset = DogCatDataset(test_imgs, class_to_int , dir_test, mode="test", transforms=get_train_transform())

#print(train_dataset[0])
#print(val_dataset[0])
if __name__ == '__main__':
    train_data_loader = DataLoader(
        dataset = train_dataset,
        num_workers = 1,
        batch_size = 32,
        shuffle = True
    )

    val_data_loader = DataLoader (
        dataset= val_dataset,
        num_workers = 1,
        batch_size = 32,
        shuffle = True
    )

    test_dataloader = DataLoader (
        dataset = test_dataset,
        num_workers = 1,
        batch_size = 32,
        shuffle = True
    )

#

    # Logs - Helpful for plotting after training finishes
    train_logs = {"loss" : [], "accuracy" : [], "time" : []}
    val_logs = {"loss" : [], "accuracy" : [], "time" : []}

    # Loading model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    my_model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(my_model.parameters(), lr = 0.0001)

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

    #Loss Function
    criterion = nn.BCELoss()

    train(model=my_model,epochs=args.epochs,optimizer=optimizer,criterion=criterion,scheduler=lr_scheduler, device=device, train_data_loader=train_data_loader, val_data_loader=val_data_loader)