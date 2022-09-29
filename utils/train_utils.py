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

from utils.utils import *

def train_one_epoch(train_data_loader,device, model, optimizer, criterion):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    

    ###Iterating over data loader
    for images, labels in train_data_loader:

        print(images)
        print(labels)

        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        _loss.backward()
        optimizer.step()
    
    # ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    # train_logs["loss"].append(epoch_loss)
    # train_logs["accuracy"].append(epoch_acc)
    # train_logs["time"].append(total_time)
        
    return epoch_loss, epoch_acc, total_time

def val_one_epoch(val_data_loader, best_val_acc, device, model, optimizer, criterion):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in val_data_loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    # val_logs["loss"].append(epoch_loss)
    # val_logs["accuracy"].append(epoch_acc)
    # val_logs["time"].append(total_time)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),"resnet50_best.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc

def train(model,epochs,optimizer,criterion,scheduler, device, train_data_loader, val_data_loader):
    best_val_acc = 0
    
    for epoch in range(epochs):
        
        ###Training
        loss, acc, _time = train_one_epoch(train_data_loader, device, model, optimizer, criterion)
        
        #Print Epoch Details
        print("\nTraining")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))
        
        ###Validation
        loss, acc, _time, best_val_acc = val_one_epoch(val_data_loader, best_val_acc, device, model, optimizer, criterion)
        
        #Print Epoch Details
        print("\nValidating")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))

    torch.save(model.state_dict(), "trained_model.pth")