## IMPORTS ##
from models.DecrypterModel import DecrypterModel
from models.UnetModel import UNet
import torch
import numpy as np

## FROM ##
from classes.LoadImages import *
from utils import *
from torch.utils.data.dataloader import DataLoader
from torchvision import models


## Functions ##

def loadData(train_path: str, train_original_path: str):
    batch_size = 4
    dataset = LoadImages(train_path, train_original_path)
    print(len(dataset))
    #test, _ = torch.utils.data.random_split(dataset, [3, len(dataset) - 3])
    #return DataLoader(test, batch_size, shuffle=True)
    return DataLoader(dataset, batch_size, shuffle=True)

torch.manual_seed(1234)

D_out = 3*96*96     # output dimension

train_loader = loadData("dataset/1B/train_1B_tiny.npy", "dataset/original/train_original_tiny.npy")
test_loader = loadData("dataset/1B/test_1B.npy", "dataset/original/test_original.npy")
print(len(test_loader))
# batch shape : (64, 3, 96, 96)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = DecrypterModel(D_out)
model = UNet(3,3,False)
# model = models.resnet18()
# model.fc = torch.nn.Linear(512, D_out)
# model = models.resnet50()
# model.fc = torch.nn.Sequential(
#     torch.nn.Linear(2048, 1024),
#     torch.nn.ReLU(),
#     torch.nn.Linear(1024,512),
#     torch.nn.ReLU(),
#     torch.nn.Linear(512, 10)
# )
#print(model.conv1)

#freeze_model(model)

train_optim(model, train_loader, test_loader, epochs=5, log_frequency=1, device=device)
torch.save(model.state_dict(), 'app/trained_model/unet_1B')
