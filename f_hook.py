import torch 
import torchvision
import numpy as np 
import pdb
import torch.optim as optim
import torch.nn as nn
import glob
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torchvision.transforms import RandomResizedCrop, ToTensor, Normalize, RandomRotation, RandomHorizontalFlip
import torchvision.models as models
import pandas as pd 
import torchvision.datasets as datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def load_dataset(batch_size = 64):
    data_path = 'autism_faces/train/'
    valid_path = 'autism_faces/valid/'
    test_path = 'autism_faces/test/'
    transformed = transforms.Compose([RandomResizedCrop(224),RandomHorizontalFlip(),  RandomRotation(10), ToTensor() ])
    train_dataset = ImageFolderWithPaths(
        root=data_path,
        transform=transformed
    )
    validation_dataset = ImageFolderWithPaths(
        root=valid_path,
        transform=transformed
    )
    test_dataset=ImageFolderWithPaths(
        root=test_path,
        transform=transformed
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True, 
        drop_last = False
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=100,
        num_workers=0,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=300,
        num_workers=0,
        shuffle=True,
        drop_last=False
    )
    return train_loader, validation_loader, test_loader
epochs = 40
network = models.resnet18(pretrained=True)
network.fc = nn.Sequential(nn.Linear(network.fc.in_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, 256),
                                  nn.Linear(256, 100),
                                  nn.Linear(100,2))
f = torch.load('../Desktop/model_resnet18.pth', map_location=torch.device('cpu'))
network.load_state_dict(f)
neuron_output = {}
def get_neuron_output(name):
    def hook(model, input, output):
        neuron_output[name] = output.detach()
    return hook

for (i, n) in network.named_modules():
    if i == 'fc.4': 
        n.register_forward_hook(get_neuron_output(i))


train_load, validation_load, test_load = load_dataset()
clusters = None
ids = train_load.dataset.samples
with torch.no_grad(): 
    for idx, (data, target, paths) in enumerate(train_load): 
        x = data
        output = network(x)
        out = neuron_output['fc.4'].numpy()
        if clusters is None: 
            clusters = pd.DataFrame(out, columns=range(1,101), index = list(paths))
        else: 
            temp = pd.DataFrame(out, columns=range(1,101), index = list(paths))
            clusters = pd.concat([clusters,temp])
    for idx, (data_val, target_val, paths) in enumerate(validation_load):
        x = data_val
        output = network(x)
        out = neuron_output['fc.4'].numpy()
        temp = pd.DataFrame(out, columns=range(1,101), index = list(paths))
        clusters = pd.concat([clusters,temp])
    for idx, (data, target, paths) in enumerate(test_load): 
        x = data
        output = network(x)
        out = neuron_output['fc.4'].numpy()
        temp = pd.DataFrame(out, columns=range(1,101), index = list(paths))
        clusters = pd.concat([clusters,temp])
    clusters.to_csv('output.csv')