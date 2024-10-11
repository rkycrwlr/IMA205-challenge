import torch
import torch.nn as nn
from torchvision import transforms, models

class EnsembleModelResNet(nn.Module):
    """
    This class defines a model that takes as input the lesion image and the features extracted by hand to predict the CLASS
    It uses a ResNet101 as deep features extractor
    """
    def __init__(self):
        super().__init__()
        # Resnet101 is used as base model and will be finetuned on our data
        model_res = models.resnet101(weights='IMAGENET1K_V2')
        # Remove the last fully connected layer
        model_res_no_fc = torch.nn.Sequential(*(list(model_res.children())[:-1]))
        self.cnn = model_res_no_fc
        self.lin1 = nn.Linear(2091, 128)

    def forward(self, x1, x2):
        # CNN to extract deep neural network features
        x1 = self.cnn(x1)
        # Concat CNN features and extracted features
        emb = torch.cat((x1,x2.unsqueeze(-1).unsqueeze(-1)), dim=1)
        # Fully connected layer to predict the final CLASS
        out = self.lin1(emb.view(emb.shape[0],-1))
        return out


class EnsembleModelEffNet(nn.Module):
    """
    This class defines a model that takes as input the lesion image and the features extracted by hand to predict the CLASS
    It uses a EfficientNet_B0 as deep features extractor
    """
    def __init__(self):
        super().__init__()
        # EfficientNet_b0 is used as base model and will be finetuned on our data
        model_eff = models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Remove the last fully connected layer
        model_eff_no_fc = torch.nn.Sequential(*(list(model_eff.children())[:-1]))
        self.cnn = model_eff_no_fc
        self.drop = nn.Dropout(p=0.4, inplace=True)
        self.lin1 = nn.Linear(1323, 8)

    def forward(self, x1, x2):
        # CNN to extract deep neural network features
        x1 = self.cnn(x1)
        x1 = self.drop(x1)
        # Concat CNN features and hand made features
        emb = torch.cat((x1,x2.unsqueeze(-1).unsqueeze(-1)), dim=1)
        # Fully connected layer to predict the final CLASS
        out = self.lin1(emb.view(emb.shape[0],-1))
        return out

class EnsembleModel(nn.Module):
    """
    This class defines a model that takes as input the lesion image and the features extracted by hand to predict the CLASS
    It uses two CNNs as feature extractors 
    """
    def __init__(self, cnn1, cnn2):
        super().__init__()
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.lin = nn.Linear(16,8)

    def forward(self, x1, x2):
        with torch.no_grad():
            emb1 = self.cnn1(x1,x2)
            emb2 = self.cnn2(x1,x2)
        emb = torch.cat((emb1,emb2), dim=1)
        out = self.lin(emb)
        return out