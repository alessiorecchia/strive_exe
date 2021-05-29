import matplotlib.pyplot as plt

import torch
import numpy as np

from torch import optim
from torchvision import datasets, transforms

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


# class_list = train_data.classes

def view_classify_general(img, ps, class_list):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    imshow(img, ax=ax1, normalize=True)
    ax1.axis('off')
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels([x for x in class_list], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

train_transf = transforms.Compose([transforms.Resize((28, 28)),
                                       transforms.RandomRotation(30),
                                    #    transforms.RandomResizedCrop(28),
                                    #    transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

test_transf = transforms.Compose([transforms.Resize((28, 28)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])


data_dir = "../../../../Datasets/cat_vs_dog" # or the path where you have downloaded the dataset

train_set = datasets.ImageFolder(data_dir + '/train', transform=train_transf)
test_set = datasets.ImageFolder(data_dir + '/test', transform=test_transf)

trainloader_ex =  torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
testloader_ex  = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Exercise_net(nn.Module):
    def __init__(self):
        super(Exercise_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*26*26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        return x

model = Exercise_net()

class_list = train_set.classes

images, labels = next(iter(trainloader_ex))
img, label = images[0], labels[0]
# Flatten images
# Forward pass, get our logits
logits = model(img.view(1, *images[0].shape))
# Calculate the loss with the logits and the labels
# ps = F.softmax(torch.exp(logits), dim=1)
# ps = torch.exp(logits)
ps = F.softmax(logits)
    
view_classify_general(img, ps, class_list)

# First step: defining criterion and optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# criterion = nn.NLLLoss()


# first step for testing pourposes

print('Initial weights - ', model.fc1.weight)

images, labels = next(iter(trainloader_ex))
# images.resize_(trainloader_ex.batch_size, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model.fc1.weight.grad)
optimizer.step()

def check_accuracy(test_loader, model):
    acc_list = []
    y_preds_list = []
    y_true_list = []
    for i, (images_test, y_true) in enumerate(iter(test_loader)):
        y_preds = []

        # Flatten EMNIST images into a 784 long vector
        # images_test.resize_(images_test.size()[0], 784)
        logits = model.forward(images_test)
        output_preds = F.softmax(logits, dim=1)
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
'''
epochs = 10
print_every = 40
accs_test = []

for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e+1}/{epochs}")

    for i, (images, labels) in enumerate(iter(trainloader_ex)):

        # Flatten EMNIST images into a 784 long vector
        # images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = model.forward(images)   # 1) Forward pass
        # print(output, output.shape)
        loss = criterion(output, labels) # 2) Compute loss
        # print(loss)
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        running_loss += loss.item()
        
        if i % print_every == 0:
            print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
            running_loss = 0
    # model.eval()
    # with torch.no_grad():

    #     acc, y_pred, y_true = check_accuracy(testloader_ex, model)
    #     accs_test.append(acc)
    # model.train()
'''
# Run this to test your data loaders
# images, labels = next(iter(trainloader_ex))
# imshow(images[0], normalize=False)

from PIL import Image
img = Image.open("imgs/catsven.png")

img = img.convert('RGB')

img

trans = transforms.ToTensor()
img = trans(img)

img.resize_(1, 3, 28, 28)

logits = model(img.view(1, *images[0].shape))

ps = F.softmax(logits, dim=1)

view_classify_general(img[0], ps, class_list)