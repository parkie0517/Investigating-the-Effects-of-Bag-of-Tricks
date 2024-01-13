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
    transforms.ToTensor(), # basic
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the training dataset and create a trainloader
trainset = torchvision.datasets.CIFAR10(root='/root/datasets/ViT_practice/cifar10/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)

# Load the testing dataset and create a testloader
testset = torchvision.datasets.CIFAR10(root='/root/datasets/ViT_practice/cifar10/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


"""
3. Define the Model (ResNet-50)
"""
model = torchvision.models.resnet50(weights=None).to(device) # Use pre-defined ResNet-50 and transfer the model to the device


"""
4. Training
"""
# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # 나중에 weight decay 포함시키기
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

total_epoch = 100
train_cnt = 0
train_loss = 0.0
train_correct = 0
train_step = 0
val_cnt = 0
val_loss = 0.0
val_correct = 0
val_step = 0

# Train the network
for epoch in range(1, total_epoch+1):  # loop over the dataset multiple times
    train_cnt = 0
    train_loss = 0.0
    train_correct = 0
    train_step = 0

    for step, batch in enumerate(trainloader):
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device) # Transfer the data to the device

        optimizer.zero_grad() # initialize the grdients to zero

        outputs = model(batch[0]) # forward pass
        loss = criterion(outputs, batch[1]) # calcuate the loss according to the output of the model
        loss.backward() # calculate the gradients
        optimizer.step() # update the gradients

        train_loss += loss.item()
        _, predict = outputs.max(1)
        train_step += 1
        train_cnt += batch[1].size(0) # count the total number of data
        train_correct += predict.eq(batch[1]).sum().item()
        """
        if step % 100 == 99: # print every 100 steps   
            print(f'Epoch: {epoch} ({step}/{len(trainloader)}), Train Acc: {100.0*train_correct/train_cnt:.2f}%, Train Loss: {train_loss/train_step:.4f}')
        """
    print(f'Epoch: {epoch}, Train Acc: {100.0*train_correct/train_cnt:.2f}%, Train Loss: {train_loss/train_step:.4f}')    
    writer.add_scalar("Loss/train", train_loss/train_step, epoch)
    writer.add_scalar("Acc/train", 100.0*train_correct/train_cnt, epoch)

    val_cnt = 0
    val_loss = 0.0
    val_correct = 0
    val_step = 0

    with torch.no_grad():
        for step, batch in enumerate(trainloader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device) # Transfer the data to the device
            outputs = model(batch[0])
            loss = criterion(outputs, batch[1])

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_step += 1
            val_cnt += batch[1].size(0)
            val_correct += predicted.eq(batch[1]).sum().item()
    print(f'Epoch: {epoch}, Val Acc: {100.0*val_correct/val_cnt:.2f}%, Val Loss: {val_loss/val_step:.4f}')      
    writer.add_scalar("Loss/val", val_loss/val_step, epoch)
    writer.add_scalar("Acc/val", 100.0*val_correct/val_cnt, epoch)
    writer.flush()

    # scheduler.step()

writer.close()
print('Finished Training')
