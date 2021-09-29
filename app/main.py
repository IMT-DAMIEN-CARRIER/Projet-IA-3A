import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class LoadImages(Dataset):
    def __init__(self, transformed_dataset_path: str, original_dataset_path: str) -> None:
        super(LoadImages, self).__init__()

        self.transformed_dataset_path = transformed_dataset_path
        self.original_dataset_path = original_dataset_path

        e1 = np.load(original_dataset_path)

        self.original_dataset = np.reshape(e1, (-1, 3, 96, 96))

        e2 = np.load(transformed_dataset_path)
        self.transformed_dataset = np.reshape(e2, (-1, 3, 96, 96))

        self.size = len(self.original_dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.transformed_dataset[idx], self.original_dataset[idx])


# Nous avons modifier l'architecture des dossiers.
# Attention à bien modifier les chemins.
o_dataset = "dataset/1A/train_1A_tiny.npy"
t_dataset = "dataset/original/train_original_tiny.npy"
batch_size = 32
dataset = LoadImages(t_dataset, o_dataset)

dataloader = DataLoader(dataset, batch_size, shuffle=True)

for batch in dataloader:
    t_img, o_img = batch

    plt.figure(1)
    plt.imshow(t_img[0].permute(1, 2, 0))
    plt.figure(2)
    plt.imshow(o_img[0].permute(1, 2, 0))
    plt.show()
    plt.savefig('images/test')

    break