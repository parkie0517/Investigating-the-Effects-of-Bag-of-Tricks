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
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
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

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0] # Select the first token from each sequence, which is the classification token  
        x = self.head(x) # pass the first token into the classification head
        return x

def main():
    # argparser
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=100)
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
    model = ViT()
    model = model.to(device) # Sends the model to a selected device
    
    # Set information about the training process
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ops.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=0) # eta_min is the value that becomes the final LR

    """
        5. Training and Testing
    """
    print("training...")
    for epoch in range(1, ops.epoch+1): # From 1 ~ ops.epoch

        model.train()
        tic = time.time()
        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]

            optimizer.zero_grad()
            output = model(img)  # [N, 10]
            loss = criterion(output, target)

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
        print(f"Epoch: {epoch}, val_acc: {val_accuracy:.2f}%, val_loss: {val_avg_loss:.4f}")

        # Leanring rate scheduling
        scheduler.step()

        # Use tensorboard to record the validation acc and loss
        writer.add_scalar('Acc/val', val_accuracy, epoch) # adds val acc
        writer.add_scalar('Loss/val', val_avg_loss, epoch) # adds val loss
        writer.flush() # inclue this line so that results are properly written in the disk
    


if __name__ == '__main__':
    # Tensorboard setting
    writer = SummaryWriter('./logs/vit/basic') # Writes training results in './logs/' directory
    main()
    writer.close() # Must include this code when finish training results