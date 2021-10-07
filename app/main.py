## IMPORTS ##
from models.DecrypterModel import DecrypterModel
from models.UnetModel import UNet
import torch
#import matplotlib.pyplot as plt
import numpy as np

## FROM ##
from classes.LoadImages import *
from utils import *
from torch.utils.data.dataloader import DataLoader


## Functions ##

def loadData(train_path: str, train_original_path: str):
    batch_size = 4
    dataset = LoadImages(train_path, train_original_path)
    return DataLoader(dataset, batch_size, shuffle=True)

torch.manual_seed(1234)

D_out = 3*96*96     # output dimension

train_loader = loadData("dataset/1A/train_1A.npy", "dataset/original/train_original.npy")
test_loader = loadData("dataset/1A/test_1A.npy", "dataset/original/test_original.npy")
# batch shape : (64, 3, 96, 96)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = DecrypterModel(D_out)
model = UNet(3,3,False)

model.load_state_dict(torch.load("trained_model/unet_epoch9"))

#freeze_model(model)
evaluate_model(model, test_loader, device)

#for i in range(9,10):
#    train_optim(model, train_loader, test_loader, epochs=1, log_frequency=1, device=device)
#    torch.save(model.state_dict(), "trained_model/unet_epoch" + str(i))