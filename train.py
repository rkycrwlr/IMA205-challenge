import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import time
import os
import copy

from model import EnsembleModel, EnsembleModelResNet, EnsembleModelEffNet
from dataset import WholeDatasetWithFeatures

from constants import NUM_EPOCHS_CLS, BATCH_SIZE_CLS, DEVICE, WORKING_DIR, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH

def logits_to_class(x):
    y = nn.Softmax(dim=1)(x)
    y = torch.argmax(y, dim=1)
    return y

def training(ensemble_model, optimizer, lossFunction, out_model_name, patience_value=5):
    
    start_time = time.time()
    best_loss = float('inf')
    best_model_weights = None
    patience = patience_value

    trainSteps = len(trainDataset) / BATCH_SIZE_CLS
    valSteps = len(valDataset) / BATCH_SIZE_CLS

    # Weights of the classes in test set to compute accuracy properly
    weights = torch.tensor([0.7005531, 0.24592265, 0.95261733, 3.64804147, 1.20674543, 13.19375, 12.56547619, 5.04219745]).to(DEVICE)

    ensemble_model = ensemble_model.to(DEVICE)

    for e in tqdm(range(NUM_EPOCHS_CLS)):
        ensemble_model.train()

        totTrainLoss = 0
        totValLoss = 0
        totTrainAcc = 0
        totValAcc = 0
        tot1, tot2 = 0, 0

        for i,(x1,x2,y) in enumerate(trainLoader):
            (x1,x2,y) = (x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE))

            # Compute prediction in Forward pass
            pred = ensemble_model(x1, x2)

            # Compute loss value
            loss = lossFunction(pred, y)

            # Set grad to zero
            optimizer.zero_grad()

            # Backward Propagation
            loss.backward()
            optimizer.step()

            totTrainLoss += loss
            totTrainAcc += torch.sum((logits_to_class(pred) == y)*weights[y])
            tot1 += torch.sum(weights[y])
        
        # For evaluation, don't compute the gradients
        with torch.no_grad():
            ensemble_model.eval()

            for (x1,x2,y) in valLoader:
                (x1,x2,y) = (x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE))

                pred = ensemble_model(x1, x2)
                totValLoss += lossFunction(pred, y)
                totValAcc += torch.sum((logits_to_class(pred) == y)*weights[y])
                tot2 += torch.sum(weights[y])
        
        avgTrainLoss = (totTrainLoss / trainSteps).cpu().detach().numpy()
        avgValLoss = (totValLoss / valSteps).cpu().detach().numpy()
        totTrainAcc = (totTrainAcc / tot1).cpu().detach().numpy()
        totValAcc = (totValAcc / tot2).cpu().detach().numpy()

        # Early Stopping policy
        if avgValLoss < best_loss:
            best_loss = avgValLoss
            best_model_weights = copy.deepcopy(ensemble_model.state_dict()) # Save weights of the model with lowest loss on validation set
            patience = patience_value
        else:
            patience -= 1
            if patience <= 0:
                break

        print("EPOCH {}, train loss : {:.4f}, train acc : {:.4f}, val loss : {:.4f}, val acc : {:.4f},".format(e+1, avgTrainLoss, totTrainAcc, avgValLoss, totValAcc))

    print("Total Training Time : {}".format(time.time() - start_time))
    torch.save(best_model_weights, out_model_name)


# Data Augmentation techniques to reduce overfitting
tsfm_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),                   # Add Gaussian Blur
    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),  # Add Color Jittering
    transforms.RandomRotation(90),                                  # Rotate the image
    transforms.RandomHorizontalFlip(),                              # Flip Horizontally
    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classic tranform without data augmentation
tsfm_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
trainDataset = WholeDatasetWithFeatures(WORKING_DIR+"/Train/Train/", WORKING_DIR+"/train_features_clean.csv", transform=tsfm_train)
valDataset = WholeDatasetWithFeatures(WORKING_DIR+"/Train/Train/", WORKING_DIR+"/val_features_clean.csv", transform=tsfm_val)
testDataset = WholeDatasetWithFeatures(WORKING_DIR+"/Test/Test/", WORKING_DIR+"/test_features_clean.csv", transform=tsfm_val)

trainLoader = DataLoader(trainDataset, shuffle=True, batch_size=BATCH_SIZE_CLS)
valLoader = DataLoader(valDataset, shuffle=False, batch_size=BATCH_SIZE_CLS)
testLoader = DataLoader(testDataset, shuffle=False, batch_size=BATCH_SIZE_CLS)

### Train Resnet

# Cross Entropy loss function
ensemble_model_resnet = ensemble_model_resnet.to(DEVICE)
lossFunction = CrossEntropyLoss()

# Adam optimizer with 1e-5 learning rate for Resnet101 and 1e-3 learning rate for fully connected layer with 1e-5 weight decay regularization
optimizer = Adam(
    [
        {"params": ensemble_model_resnet.lin1.parameters(), "lr": 1e-3},
        {"params": ensemble_model_resnet.cnn.parameters(), "lr": 1e-5}
    ],
    lr=1e-5, weight_decay=1e-5
)

# Training is long even on GPU (around 20min / epoch)
training(ensemble_model_resnet, optimizer, lossFunction, "ensemble_101_balanced_cls.pth")

### Train Effnet

# Cross Entropy loss function
ensemble_model_effnet = ensemble_model_effnet.to(DEVICE)
lossFunction = CrossEntropyLoss()

# Adam optimizer with 1e-4 learning rate for EfficientNet_B0 and 1e-3 learning rate for fully connected layer with 1e-5 weight decay regularization
optimizer = Adam(
    [
        {"params": ensemble_model_effnet.lin1.parameters(), "lr": 1e-3},
        {"params": ensemble_model_effnet.cnn.parameters(), "lr": 1e-4}
    ],
    lr=1e-4, weight_decay=1e-5
)

# Training is long even on GPU (around 20min / epoch)
training(ensemble_model_effnet, optimizer, lossFunction, "ensemble_b0_balanced_cls.pth")

### Train Ensemble Model

ensemble_model_resnet = EnsembleModelResNet()
ensemble_model_resnet.load_state_dict(torch.load(WORKING_DIR+"/ensemble_101_balanced_cls.pth", map_location=DEVICE))

ensemble_model_effnet = EnsembleModelEffNet()
ensemble_model_effnet.load_state_dict(torch.load(WORKING_DIR+"/ensemble_b0_balanced_cls.pth", map_location=DEVICE))

global_model = EnsembleModel(ensemble_model_resnet, ensemble_model_effnet)
global_model = global_model.to(DEVICE)

# Cross Entropy loss function
lossFunction = CrossEntropyLoss()

# Adam optimizer with 1e-3 learning rate for the last fully connected layer with 1e-5 weight decay regularization
# We only train the last linear layer as the CNNs are already trained
optimizer = Adam(global_model.lin.parameters(), lr=1e-3, weight_decay=1e-5)

# Training is long even on GPU (around 20min / epoch)
training(ensemble_model_effnet, optimizer, lossFunction, "ensemble_glob_balanced_cls.pth")