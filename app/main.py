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
    return DataLoader(dataset, batch_size, shuffle=True)

def loadDataTest(train_path: str, train_original_path: str):
    batch_size = 1
    dataset = LoadImages(train_path, train_original_path)
    return DataLoader(dataset, batch_size, shuffle=True)

torch.manual_seed(1234)

D_out = 3*96*96     # output dimension

train_loader = loadData("dataset/1A/train_1A.npy", "dataset/original/train_original.npy")
test_loader = loadDataTest("dataset/2/test_2.npy", "dataset/original/test_original.npy")
# batch shape : (64, 3, 96, 96)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#model = DecrypterModel(D_out)

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

for i in range(13, 14):
    model = UNet(3,3,False)

    model.load_state_dict(torch.load("trained_model/unet_2_epoch"+str(i)))
    evaluate_model(model, test_loader, device)

#train_optim(model, train_loader, test_loader, epochs=5, log_frequency=1, device=device)

#for i in range(10,13):
#    train_optim(model, train_loader, test_loader, epochs=1, log_frequency=1, device=device)
#    torch.save(model.state_dict(), "trained_model/unet_epoch" + str(i))