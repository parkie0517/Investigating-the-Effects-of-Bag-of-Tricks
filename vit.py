"""
    1. Import Libraries and do some settings
"""
import os
import time
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from torchvision.datasets.cifar import CIFAR10
from tensorboardX import SummaryWriter
import numpy as np # Added numpy for mixup


"""
    2. Define the ViT Model
"""
# Define the Embedding Layer
class EmbeddingLayer(nn.Module):
    def __init__(self, in_chans, embed_dim, img_size, patch_size):
        super().__init__()
        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # Using Conv2d operation to perform linear projection

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Create a classification token (used later for classifying the image)
        self.num_tokens += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim)) # Create positional embedding parameters

        nn.init.normal_(self.cls_token, std=1e-6) # Initialize classfication token using normal(Gaussian) distibution
        trunc_normal_(self.pos_embed, std=.02) # Initialize positional embeddings using truncated normal distribution

    def forward(self, x): # Define the forward function
        B, C, H, W = x.shape
        embedding = self.project(x) # Perform Linear projection (=tokenization of the image)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC

        # Add the classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls_tokens, z], dim=1)

        # Add the position embedding
        z = z + self.pos_embed
        return z

# Define the Multi-Head Self-Attention Layer
class MSA(nn.Module):
    def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # define the scaling scalar

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # define the matrices weight_queue,  weight_key,  weight_value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # create the queue, key, value
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # apply drop out to the attention scores

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Define the MLP layer
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias) # define the first mlp layer
        self.act = act_layer() # define the activation layer
        self.drop1 = nn.Dropout(drop) # define the first dropout layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias) # define the second mlp layer
        self.drop2 = nn.Dropout(drop) # define the second dropout layer

    def forward(self, x): # define the forard function
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Define the Encoder Block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim) # define the first layer normalization layer (uses layer normalization instead of batch normalization)
        self.norm2 = norm_layer(dim) # define the second layer normalization layer
        self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x): # define the forward pass
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Define the ViT model
class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # final norm
        self.norm = norm_layer(embed_dim)

        # Define the Classification Head
        # Reduce the dimension to num of classes for classifying
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def apply_zero_init(self, first=False, second=False):
        for block in self.blocks.children():
            if first:
                nn.init.zeros_(block.norm1.weight) # applies zero init to the first LN layer
            if second:
                nn.init.zeros_(block.norm2.weight) # applies zero init to the second LN layer

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0] # Selects the first column from x, which is the classification token  
        x = self.head(x) # pass the first token into the classification head
        return x

class LabelSmoothingCrossEntropy(nn.Module):
    """
        This is an implementation of a custom label smoothing code
    """
    def __init__(self, smoothing=0.1): # smoothing is the hyperparameter that adjusts the smoothing strength
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * (self.smoothing / (input.size(-1) - 1))
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
class WarmupScheduler:
    """
        This is an implementation of a custom learning rate warmup code
    """
    def __init__(self, optimizer, warmup_epochs, initial_lr, peak_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = ((self.peak_lr - self.initial_lr) / self.warmup_epochs) * self.current_epoch + self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
        Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam # Returns mixed inputs, pairs of targets, and lambda values

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
        Loss function used for mixup
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def calculate_mean_gamma(model):
    gamma_values = []
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            gamma_values.append(module.weight.data.mean().item())
    return np.mean(gamma_values)

def main():
    # argparser
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=90)
    parer.add_argument('--batch_size', type=int, default=256)
    parer.add_argument('--lr', type=float, default=0.001) # this is the same as 1e-3
    parer.add_argument('--name', type=str, default='vit_cifar10')
    ops = parer.parse_args()

    # Cuda setting
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    print(f'currently using {device}')

    

    """
        3. Load CIFAR10 dataset
    """
    transform_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    train_set = CIFAR10(root='/root/datasets/ViT_practice/cifar10/', train=True, download=True, transform=transform_cifar)
    test_set = CIFAR10(root='/root/datasets/ViT_practice/cifar10/', train=False, download=True, transform=test_transform_cifar)
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=ops.batch_size, num_workers = 16)
    test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=ops.batch_size, num_workers=16)

    """
        4. Load the model (already defined above)
    """
    # Create the model instance
    model = ViT(drop_rate=0.0, attn_drop_rate=0.0) # Applies dropout in the MLP and MHSA
    model.apply_zero_init(True, False) # Applies zero initialization to the ViT model
    model = model.to(device) # Sends the model to a selected device
    
    # Set information about the training process
    """
        5. Choosing the loss function
            - Cross entropy loss
            - Label smoothing cross entropy loss
    """
    # criterion = nn.CrossEntropyLoss() # Cross entropy loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1) # Label smoothing loss

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ops.lr, weight_decay=5e-5)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=0) # eta_min is the value that becomes the final LR
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, initial_lr=1e-6, peak_lr=ops.lr)
    """
        6. Training and Testing
    """
    print("training...")
    for epoch in range(1, ops.epoch+1): # From 1 ~ ops.epoch
        # Leanring rate scheduling
        if epoch <= 5:
            warmup_scheduler.step()
        else:
            # After the warmup period, apply the cosine annealing scheduler
            scheduler.step()

        model.train()
        tic = time.time()
        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]
            img, target_a, target_b, lam = mixup_data(img, target, alpha=1.0, use_cuda=True)

            optimizer.zero_grad()
            output = model(img)  # [N, 10]
            # loss = criterion(output, target) # ordinary cross entropy loss
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)

            loss.backward() # Calculates the gradients for updates
            optimizer.step() # Updates the parameters

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

        # Test the model performance
        #print('Validation of epoch [{}]'.format(epoch))
        model.eval()
        correct = 0
        val_avg_loss = 0
        total = 0
        with torch.no_grad():

            for idx, (img, target) in enumerate(test_loader):
                model.eval()
                img = img.to(device)  # [N, 3, 32, 32]
                target = target.to(device)  # [N]

                output = model(img)  # [N, 10]
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                # first eval
                pred, idx_ = output.max(-1)
                correct += torch.eq(target, idx_).sum().item()
                total += target.size(0)
                val_avg_loss += loss.item()

        val_accuracy = (correct / total) * 100
        val_avg_loss = val_avg_loss / len(test_loader)
        mean_gamma = calculate_mean_gamma(model)
        print(f"Epoch: {epoch}, val_acc: {val_accuracy:.2f}%, val_loss: {val_avg_loss:.4f}, mean_gamma: {mean_gamma:.4f}")
        
        # Use tensorboard to record the validation acc and loss
        writer.add_scalar('Acc/val', val_accuracy, epoch) # adds val acc
        writer.add_scalar('Loss/val', val_avg_loss, epoch) # adds val loss
        writer.add_scalar('Misc/gamma', mean_gamma, epoch) # add the mean of the gamma value
        writer.flush() # inclue this line so that results are properly written in the disk
    


if __name__ == '__main__':
    # Tensorboard setting
    writer = SummaryWriter('./logs/vit_test/basic+cosine+warm+label+mixup+zero(first LN)') # Writes training results in './logs/' directory
    main()
    writer.close() # Must include this code when finish training results