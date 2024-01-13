"""
1. Import Libraries and do some settings
"""
import torch # import torch
import torchvision # import torchvision
import torchvision.transforms as transforms # import transforms which is used for data pre-processing
import torch.nn as nn # import torch.nn for defining and building neural networks
import torch.nn.functional as F # import functional for using the activation functions
import torch.optim as optim # import torch.optim for using optimizers
from tensorboardX import SummaryWriter # import tensorbardX which is used for visualing result 

# Tensorboard settings
writer = SummaryWriter('./logs/base') # Write training results in './logs/' directory

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
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

model = ResNet50() # Use cumtom made ResNet-50
# model = torchvision.models.resnet50(weights=None).to(device) # Use pre-defined ResNet-50 
model = model.to(device) # Transfer the model to the device

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
    
    # Write validation results to tensorboard
    writer.add_scalar("Loss/train_1", train_loss/train_step, epoch)
    writer.add_scalar("Acc/train_1", 100.0*train_correct/train_cnt, epoch)

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
    
    # Write validation results to tensorboard
    writer.add_scalar("Loss/val_1", val_loss/val_step, epoch)
    writer.add_scalar("Acc/val_1", 100.0*val_correct/val_cnt, epoch)
    
    writer.flush() # make sure the results are written properly into the storage

    # scheduler.step()

writer.close() # close writing the results to the storage
print('Finished Training')
