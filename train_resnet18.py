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

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)  

writer = SummaryWriter('run/inception_run')

def load_dataset(batch_size = 64):
    data_path = 'autism_faces/train/'
    valid_path = 'autism_faces/valid/'
    test_path = 'autism_faces/test/'
    transformed = transforms.Compose([RandomResizedCrop(224),RandomHorizontalFlip(),  RandomRotation(10), ToTensor() ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformed
    )
    validation_dataset = torchvision.datasets.ImageFolder(
        root=valid_path,
        transform=transformed
    )
    test_dataset=torchvision.datasets.ImageFolder(
        root=test_path,
        transform=transformed
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=100,
        num_workers=0,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
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
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
loss = nn.CrossEntropyLoss() 
train_load, validation_load, test_load = load_dataset()
for epoch in range(epochs): 
    network.train()
    total = 0
    correct = 0
    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_load):
        with torch.cuda.device(dev): 
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_predicted = network(data)
            err = loss(y_predicted, target) 
            writer.add_scalar('training loss', err, (epoch * len(train_load))+batch_idx)
            err.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).sum()
            target = target.to('cpu')
            predicted = predicted.to('cpu')
            total_train_loss+=err.data
        tn, fp, fn, tp = confusion_matrix(predicted, target).ravel()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_train = 2*((precision+recall)/(precision*recall))
        writer.add_scalar('training precision', precision, (epoch * len(train_load))+batch_idx)
        writer.add_scalar('training recall', recall, (epoch * len(train_load))+batch_idx)
        writer.add_scalar('training f1', f1_train, (epoch * len(train_load))+batch_idx)
    accuracy = 100. * float(correct) / total
    writer.add_scalar('training accuracy',accuracy, (epoch * len(train_load))+batch_idx)
    print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (batch_idx + 1), accuracy))
    network.eval()
    with torch.no_grad(): 
        total = 0 
        correct = 0 
        total_validation_loss = 0
        for idx, (data_val, target_val) in enumerate(validation_load):
            with torch.cuda.device(dev): 
                data_val = data.to(device)
                target_val = target.to(device)
                target_val_predicted = network.forward(data_val)
                v_err = loss(target_val_predicted, target_val)
                writer.add_scalar('validation loss', v_err, (epoch * len(train_load))+batch_idx)
                _, predicted_val = torch.max(target_val_predicted.data, 1)
                total += target_val.size(0)
                correct += predicted_val.eq(target_val.data).sum()
                target_val = target_val.to('cpu')
                predicted_val = predicted_val.to('cpu')
                total_validation_loss+=v_err.data
            tn, fp, fn, tp = confusion_matrix(target_val, predicted_val).ravel()
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_val = 2*((precision+recall)/(precision*recall))
            writer.add_scalar('validation precision', precision, (epoch * len(train_load))+batch_idx)
            writer.add_scalar('validation recall', recall, (epoch * len(train_load))+batch_idx)
            writer.add_scalar('validation f1', f1_val, (epoch * len(train_load))+batch_idx)
        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] validation Loss: %.4f, Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (idx + 1), accuracy))
        writer.add_scalar('validation accuracy',accuracy, (epoch * len(train_load))+batch_idx)
    if epoch%5==0: 
        torch.save(network.state_dict(),'model_inception.pth')
torch.save(network.state_dict(),'model_inception.pth')
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)
writer.close()
