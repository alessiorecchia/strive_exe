import matplotlib.pyplot as plt
import time
import numpy as np

import torch.nn as nn
import torch
from torch.nn import Conv2d,  MaxPool2d, ReLU, Sequential, BatchNorm2d, Dropout, Module, Linear
from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Grayscale
import torch.functional as F


data_dir = "./MNIST_jpg" # or the path where you have downloaded the dataset



train_transform = transforms.Compose([transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])

test_transform = transforms.Compose([transforms.Grayscale(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

data_dir = 'MNIST_jpg'                                                       

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

trainloader =  torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

class_list = train_data.classes



net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.NLLLoss()

# images, labels = next(iter(trainloader))
# img, label = images[0], labels[0]
# logits = net(img.view(1, *images[0].shape))
# # Calculate the loss with the logits and the labels
# # ps = F.softmax(torch.exp(logits), dim=1)
# # ps = torch.exp(logits)
# ps = F.log_softmax(logits, dim=1)
    
# view_classify_general(img, ps, class_list)

def check_accuracy(test_loader, model):
    acc_list = []
    y_preds_list = []
    y_true_list = []
    for i, (images_test, y_true) in enumerate(iter(test_loader)):
        y_preds = []

        # Flatten EMNIST images into a 784 long vector
        # images_test.resize_(images_test.size()[0], 784)
        logits = model.forward(images_test)
        # output_preds = F.softmax(logits, dim=1)
        output_preds = (logits)
        for p in output_preds:
            y_preds.append(p.argmax())
        
        y_preds = np.array(y_preds)
        y_preds = torch.tensor(y_preds)

        for i in range(y_preds.size(0)):
            y_preds_list.append(y_preds[i].item())
            y_true_list.append(y_true[i].item())

    accuracy = (np.array(y_preds_list) == np.array(y_true_list)).sum()/len(y_preds_list)
    print(accuracy)

    return accuracy, y_preds_list, y_true_list

epochs = 5
print_every = 40
accs_test = []

start = time.time()
for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e+1}/{epochs}")

    for i, (images, labels) in enumerate(iter(trainloader)):

        # Flatten EMNIST images into a 784 long vector
        # images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = net.forward(images)   # 1) Forward pass
        # print(output, output.shape)
        loss = criterion(output, labels) # 2) Compute loss
        # print(loss)
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        running_loss += loss.item()
        
        if i % print_every == 0:
            print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
            running_loss = 0
    net.eval()
    with torch.no_grad():

        acc, y_pred, y_true = check_accuracy(testloader, net)
        accs_test.append(acc)
    net.train()
print(f'It took {time.time() - start} s to train')
