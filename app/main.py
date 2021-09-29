## IMPORTS ##
import torch
import matplotlib.pyplot as plt

## FROM ##
from classes.LoadImages import *

## Functions ##

def loadData(train_path: str, train_original_path: str):
    batch_size = 64
    dataset = LoadImages(train_path, train_original_path)

    return DataLoader(dataset, batch_size, shuffle=True)

    #for batch in dataloader:
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


data_loader = loadData("dataset/1A/train_1A_tiny.npy","dataset/original/train_original_tiny.npy")