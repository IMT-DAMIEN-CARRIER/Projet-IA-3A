## IMPORTS ##
from models.DecrypterModel import DecrypterModel
import torch
#import matplotlib.pyplot as plt
import numpy as np

## FROM ##
from classes.LoadImages import *
from utils import *
from torch.utils.data.dataloader import DataLoader


## Functions ##

def loadData(train_path: str, train_original_path: str):
    batch_size = 64
    dataset = LoadImages(train_path, train_original_path)
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
#test_loader = loadData("dataset/1A/test_1A.npy", "dataset/original/test_original.npy")
# batch shape : (64, 3, 96, 96)

train, test = iter(train_loader).next()
print(train.shape)
# print(test.shape)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DecrypterModel(D_out)

train_optim(model, train_loader, None, epochs=1, log_frequency=1, device=device)