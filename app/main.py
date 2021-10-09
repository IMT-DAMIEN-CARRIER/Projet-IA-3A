## IMPORTS ##
from models.UnetModel import UNet
import torch
import numpy as np

## FROM ##
from utils import *

# Fonction d'entrainement du model
def train_model(model, epoch) :
    train_optim(model, train_loader, test_loader, epochs=epoch, log_frequency=1, device=device)
    # A modifier pour charger le modèle de votre choix.
    torch.save(model.state_dict(), "app/trained_model/my_model")

torch.manual_seed(1234)

D_out = 3*96*96     # output dimension

# Modifier le path en fonction du dataset que vous souhaitez utiliser.
train_loader = loadData("dataset/1B/train_1B.npy", "dataset/original/train_original.npy", batch_size=4)
test_loader = loadData("dataset/1B/test_1B.npy", "dataset/original/test_original.npy")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


######## MODELS ########

model = models.resnet18()
model.fc = torch.nn.Linear(512, D_out)
#model = UNet(3,3,False)

# A decommenter pour utiliser un model enregistré.
# model.load_state_dict(torch.load("app/trained_model/unet_1B_epoch7"))


# A decommenter pour entrainer sur plusieurs epochs
epoch = 1
train_model(model, epoch=epoch)

# A décommenter pour uniquement evaluer les models
# L'evaluation est disponible dans un fichier de log sous "app/trained_model/log_model.txt"
# evaluate_model(model, test_loader, device)