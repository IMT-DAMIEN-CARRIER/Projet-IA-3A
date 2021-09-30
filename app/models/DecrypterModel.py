import torch
from torchvision import models

class DecrypterModel(torch.nn.Module):
    def __init__(self, d_out):
        super(DecrypterModel, self).__init__()
        self.model = models.vgg11()
        self.model.classifier = torch.nn.Sequential(*[torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Dropout(p=0.5),
                                                    torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Dropout(p=0.5),
                                                    torch.nn.Linear(in_features=2048, out_features=d_out, bias=True)])

    def forward(self, x):
        return self.model(x)