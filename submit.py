import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms

from model import EnsembleModel, EnsembleModelResNet, EnsembleModelEffNet
from dataset import WholeDatasetWithFeatures
from constants import WORKING_DIR, DEVICE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT

def logits_to_class(x):
    y = nn.Softmax(dim=1)(x)
    y = torch.argmax(y, dim=1)
    return y

def submit_csv(name,y_final):
    df = pd.read_csv(WORKING_DIR+"/SampleSubmission.csv")
    y_final = y_final.astype(np.uint8)
    df["CLASS"] = y_final
    df.to_csv(f"{name}.csv", index=False)


tsfm_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

testDataset = WholeDatasetWithFeatures(WORKING_DIR+"/Test/Test/", WORKING_DIR+"/test_features_clean.csv", transform=tsfm_val)

global_model = EnsembleModel(EnsembleModelResNet(), EnsembleModelEffNet())
global_model.load_state_dict(torch.load(WORKING_DIR+"/ensemble_glob_balanced_cls.pth", map_location=DEVICE))
global_model = global_model.to(DEVICE)

y_final = np.zeros((len(testDataset),))
with torch.no_grad():
    global_model.eval()
    for i,(x1,x2) in enumerate(testDataset):
        x1 = x1.unsqueeze(0).to(DEVICE)
        x2 = x2.unsqueeze(0).to(DEVICE)
        pred = nn.Softmax(dim=1)(global_model(x1,x2))
        y_final[i] = logits_to_class(pred)+1
submit_csv(WORKING_DIR + "/Submission18", y_final)