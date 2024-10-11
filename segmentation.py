import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm
import time
import numpy as np
import os
import copy
from PIL import Image
import cv2

from unet import UNet
from dataset import SegmentationDataset
from constants import WORKING_DIR, DEVICE, NUM_EPOCHS_SEG, BATCH_SIZE_SEG, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH

def dullrazor(img):
    img = np.array(img)
    img = img.astype(np.uint8)
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    #Black hat filter
    kernel = cv2.getStructuringElement(1,(9,9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    #Gaussian filter
    bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
    #Binary thresholding
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    #Replace pixels of the mask
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)
    return dst


tsfm = transforms.Compose([
    transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()
    ])

tsfm2 = transforms.Compose([
    transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
    transforms.Lambda(dullrazor), # We add the dullrazor algorithm for pre processing (long to run on a lot of files)
	transforms.ToTensor()
    ])

segtrainDataset = SegmentationDataset(WORKING_DIR+"/Train/Train/", WORKING_DIR+"/metadataTrain.csv", transform_img=tsfm2, transform_msk=tsfm)
segtestDataset = SegmentationDataset(WORKING_DIR+"/Test/Test/", WORKING_DIR+"/metadataTest.csv", transform_img=tsfm2, transform_msk=tsfm)

segtrainLoader = DataLoader(segtrainDataset, shuffle=True, batch_size=BATCH_SIZE_SEG)
segtestLoader = DataLoader(segtestDataset, shuffle=False, batch_size=BATCH_SIZE_SEG)


unet = UNet().to(DEVICE)

lossFunction = BCEWithLogitsLoss() # Binary Cross Entropy is used as we predict masks with 0 and 1 values only
optimizer = Adam(unet.parameters(),lr=0.001)

trainSteps = len(segtrainDataset) / BATCH_SIZE_SEG
testSteps = len(segtestDataset) / BATCH_SIZE_SEG

patience_value = 15 # This is the patience used for the early stopping technique

bestLoss = float("inf")
best_model_params = None
patience = patience_value

start_time = time.time()

for e in tqdm(range(NUM_EPOCHS_SEG)):
    unet.train()

    totTrainLoss = 0
    totTestLoss = 0

    for (x,y,_) in segtrainLoader:
        (x,y) = (x.to(DEVICE), y.to(DEVICE))

        # Compute prediction in Forward pass
        pred = unet(x)

        # Compute loss value
        loss = lossFunction(pred, y)

        # Set grad to zero
        optimizer.zero_grad()

        # Backward Propagation
        loss.backward()
        optimizer.step()

        totTrainLoss += loss
    
    with torch.no_grad():
        unet.eval()

        for (x,y) in segtestLoader:
            (x,y) = (x.to(DEVICE), y.to(DEVICE))

            pred = unet(x)
            totTestLoss += lossFunction(pred, y)
    
    avgTrainLoss = (totTrainLoss / trainSteps).cpu().detach().numpy()
    avgTestLoss = (totTestLoss / testSteps).cpu().detach().numpy()

    if avgTestLoss < bestLoss:
        bestLoss = avgTestLoss
        # We keep the model that has the lowest loss on Test set
        best_model_params = copy.deepcopy(unet.state_dict())
        patience = patience_value
    else:
        patience -= 1
        # Early Stopping : If the test loss hasn't imporoved in 15 epochs we stop the train
        if patience <= 0:
            break

    print("EPOCH {}, train loss : {:.4f}, test loss : {:.4f}".format(e+1, avgTrainLoss, avgTestLoss))

print("Total Training Time : {}".format(time.time() - start_time))

# Saving the model to use it later
torch.save(best_model_params, WORKING_DIR + "/unet_seg_200e.pth")