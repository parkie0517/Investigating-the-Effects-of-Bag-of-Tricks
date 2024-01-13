"""
1. Import Libraries and do some settings
"""
import torch # import torch
import torchvision # import torchvision
import torchvision.transforms as transforms # import transforms which is used for data pre-processing
import torch.nn as nn # import torch.nn for defining and building neural networks
import torch.optim as optim # import torch.optim for using optimizers
from tensorboardX import SummaryWriter # import tensorbardX which is used for visualing result 

# Tensorboard settings
writer = SummaryWriter('./logs/') # Write training results in './logs/' directory

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


"""
2. Load Dataset (CIFAR10)
"""
# Load and normalize CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset and create a trainloader
trainset = torchvision.datasets.CIFAR10(root='/root/datasets/ViT_practice/cifar10/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Load the testing dataset and create a testloader
testset = torchvision.datasets.CIFAR10(root='/root/datasets/ViT_practice/cifar10/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


"""
3. Define the Model (ResNet-50)
"""
net = torchvision.models.resnet50(weights=None).to(device) # Define ResNet-50



# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epoch_loss = 0.0

# Train the network
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device) # Transfer the data to the device

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    epoch_loss = epoch_loss/len(trainloader)        
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    
    writer.flush()

writer.close()
print('Finished Training')
