## IMPORTS ##
from models.UnetModel import UNet
import torch
import numpy as np

## FROM ##
from classes.LoadImages import *
from utils import *
from torch.utils.data.dataloader import DataLoader
from torchvision import models

def train_model(model, epoch) :
    train_optim(model, train_loader, test_loader, epochs=epoch, log_frequency=1, device=device)
    torch.save(model.state_dict(), "app/trained_model/my_model")

def loadDataTest(train_path: str, train_original_path: str):
    batch_size = 1
    dataset = LoadImages(train_path, train_original_path)
    return DataLoader(dataset, batch_size, shuffle=True)

torch.manual_seed(1234)
D_out = 3*96*96     # output dimension

# Modifier le path en fonction du dataset que vous souhaitez utiliser.
train_loader = loadData("dataset/1B/train_1B.npy", "dataset/original/train_original.npy", batch_size=4)
test_loader = loadData("dataset/1B/test_1B.npy", "dataset/original/test_original.npy")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = models.resnet18()
# model.fc = torch.nn.Linear(512, D_out)
model = UNet(3,3,False)

# A decommenter pour utiliser un model enregistré.
# model.load_state_dict(torch.load("app/trained_model/unet_1B_epoch7"))

# A décommenter pour evaluer les models existants
# Modifier le range pour choisir plusieurs models.
# for i in range(6, 12):
#     model.load_state_dict(torch.load("app/trained_model/unet_1A_epoch" + str(i)))
#     evaluate_model(model, test_loader, device, i)

# A decommenter pour entrainer sur plusieurs epochs
# epoch = 1
# train_model(model, epoch=epoch)
# evaluate_model(model, test_loader, device, epoch=epoch)
