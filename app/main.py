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
    batch_size = 8
    dataset = LoadImages(train_path, train_original_path)
    print(len(dataset))
    #test, _ = torch.utils.data.random_split(dataset, [3, len(dataset) - 3])
    #return DataLoader(test, batch_size, shuffle=True)
    return DataLoader(dataset, batch_size, shuffle=True)

    # for batch in dataloader:
    #    t_img, o_img = batch
    #
    #    plt.figure(1)
    #    plt.imshow(t_img[0].permute(1, 2, 0))
    #    plt.figure(2)
    #    plt.imshow(o_img[0].permute(1, 2, 0))
    #    plt.show()
    #    plt.savefig('images/test')
    #
    #    break

torch.manual_seed(1234)

D_out = 3*96*96     # output dimension

train_loader = loadData("dataset/1A/train_1A_tiny.npy", "dataset/original/train_original_tiny.npy")
print(len(train_loader))


test_loader = loadData("dataset/1A/test_1A.npy", "dataset/original/test_original.npy")
# batch shape : (64, 3, 96, 96)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = DecrypterModel(D_out)
model = UNet(3,D_out,False)

train_optim(model, train_loader, test_loader, epochs=1, log_frequency=1, device=device)
