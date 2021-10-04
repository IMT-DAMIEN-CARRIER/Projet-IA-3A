 import torch.nn as nn

class MyModel(torch.nn.Module):
  def __init__(self, d_out):
    super(MyModel, self).__init__()

    self.flatten = nn.Flatten()
    self.model_cnn = torch.nn.Sequential(
        *[
          nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5), stride=1, padding=2),
          nn.ReLU(),
          nn.AvgPool2d(kernel_size=(2,2), stride=2),
          nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5), stride=1),
          nn.ReLU(),
          nn.AvgPool2d(kernel_size=(2,2), stride=2),
          nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5), stride=1),
          nn.Sigmoid(),
          nn.Flatten(),
          nn.Linear(in_features=972, out_features=d_out, bias=True)
        ])

  def forward(self, x):
    return self.model_cnn(x) + self.flatten(x)