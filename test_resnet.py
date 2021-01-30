import torch 
import torchvision
import numpy as np 
import pdb
import torch.optim as optim
import model
import torch.nn as nn
import glob
import torchvision.models as models
import pdb


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)  

def load_dataset():
    test_path = 'autism_faces/test/'
    test_dataset=torchvision.datasets.ImageFolder(
        root=test_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=300,
        num_workers=0,
        shuffle=True
    )
    return test_loader
epochs = 40
network = models.resnet18(pretrained=True)
network.fc = nn.Sequential(nn.Linear(network.fc.in_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, 256),
                                  nn.Linear(256, 100),
                                  nn.Linear(100,2))
f = torch.load('../Desktop/model_resnet18.pth')
network.load_state_dict(f)
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
loss = nn.CrossEntropyLoss() 
test_load = load_dataset()
total = 0 
correct = 0 
test_loss = 0 
total = 0
network.eval()
with torch.no_grad(): 
    for idx, (data_test, target_test) in enumerate(test_load):
        with torch.cuda.device(dev): 
            data_test = data_test.to(device)
            target_test = target_test.to(device)
            target_test_predicted = network(data_test)
            T_err = loss(target_test_predicted, target_test)
            _, predicted = torch.max(target_test_predicted.data, 1)
            total += target_test.size(0)
            correct += predicted.eq(target_test.data).sum()
            print(correct)
    accuracy = 100. * float(correct) / total
    print(accuracy)

