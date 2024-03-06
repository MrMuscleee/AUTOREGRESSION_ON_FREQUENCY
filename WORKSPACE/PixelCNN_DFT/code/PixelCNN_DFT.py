#!/usr/bin/env python
# coding: utf-8

# Libraries

# In[67]:


## Standard libraries
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.fft as fft
import cv2
from PIL import Image

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

#3 Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

## Progress bar
from tqdm.notebook import tqdm




# Constants

# In[68]:


# Training tag
Train_tag = "PixelCNN_DFT_train_epoch_246_745_lr_1em4"
# Whether or not start training from previous checkpoints
Resume_from_checkpoint = True
# Path to the checkpoint
RESUME_PATH = "../checkpoints/PixelCNN_DFT_train_epoch_196_245_lr_2em4/epoch_245.ckpt"
# Path to the folder where the datasets are stored
DATASET_PATH = "../data"
# Path to the folder where the checkpoints are saved
CHECKPOINT_PATH = "../checkpoints"
# Path to the folder where the training logs are saved
LOG_PATH = "../tensorboard_log"
# Fetching the device
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)
# Learning rate
Learning_rate = 1e-4
# Max epoch
Max_epoch = 500


# Define DFT transform

# In[69]:


#helper functions
def get_centered_dft(img):
    dft = np.fft.fft2(img)
    centered_dft = np.fft.fftshift(dft)
    return centered_dft

def restore_img_from_centered_dft(centered_dft):
    dft = np.fft.fftshift(centered_dft)
    img = np.fft.ifft2(dft)
    return img

def flat_transform(dft):
    real = dft.real
    imag = dft.imag
    H, W = dft.shape
    flat_dft = np.zeros((H, 2 * W))
    flat_dft[:, ::2] = real
    flat_dft[:, 1::2] = imag
    return flat_dft

def inverse_flat_transform(flat_dft):
    real = flat_dft[:, ::2]
    imag = flat_dft[:, 1::2]
    dft = np.vectorize(complex)(real, imag)
    return dft
    

def show_centered_dft(centered_dft):
    figures_dft_phase = np.angle(centered_dft)
    figures_dft_magnitude = np.abs(centered_dft)
    figures_dft_log_magnitude = np.log10(figures_dft_magnitude)

    phase_stretched = (figures_dft_phase - np.min(figures_dft_phase)) / (np.max(figures_dft_phase) - np.min(figures_dft_phase))
    log_magnitude_stretched = (figures_dft_log_magnitude - np.min(figures_dft_log_magnitude)) / (np.max(figures_dft_log_magnitude) - np.min(figures_dft_log_magnitude))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(phase_stretched, cmap='gray')
    axs[0].set_title('Phase Component')
    axs[1].imshow(log_magnitude_stretched, cmap='gray')
    axs[1].set_title('Logarithm of Magnitude Component')
    plt.show()

def center_transform(image):
    h, w = image.shape[:2]
    n = min(h * w, min(h, w) ** 2)
    center = (h - 1) // 2, (w - 1) // 2
    indices = [(i, j) for i in range(h) for j in range(w)]
    original_indices = [(i, j) for i in range(h) for j in range(w)]
    indices.sort(key=lambda x: (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
    indices = indices[:n]
    new_image = np.zeros_like(image)
    for i in range(n):
        new_image[original_indices[i]] = image[indices[i]]
    return new_image

def inverse_center_transform(image):
    h, w = image.shape[:2]
    n = min(h * w, min(h, w) ** 2)
    center = (h - 1) // 2, (w - 1) // 2
    indices = [(i, j) for i in range(h) for j in range(w)]
    original_indices = [(i, j) for i in range(h) for j in range(w)]
    indices.sort(key=lambda x: (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
    indices = indices[:n]
    new_image = np.zeros_like(image)
    for i in range(n):
        new_image[indices[i]] = image[original_indices[i]]
    return new_image

def DFT_transform(img):
    result = torch.zeros((img.shape[0], img.shape[1], img.shape[2]*2))
    for i in range(result.shape[0]): # for every channel
        temp = img[i]
        temp = get_centered_dft(temp)
        temp = center_transform(temp)
        temp = flat_transform(temp)
        result[i] = torch.Tensor(temp)
    return result

def inverse_DFT_transform(img):
    result = torch.zeros((img.shape[0], img.shape[1], img.shape[2]//2))
    for i in range(result.shape[0]): # for every channel
        temp = img[i]
        temp = inverse_flat_transform(temp)
        temp = inverse_center_transform(temp)
        temp = restore_img_from_centered_dft(temp)
        result[i] = torch.Tensor(temp)
    return result.real


# Data

# In[70]:


# Convert images from 0-1 to 0-255 (integers). We use the long datatype as we will use the images as labels as well
def discretize(sample):
    return (sample * 255).to(torch.long)

# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([transforms.ToTensor(),
                                discretize])

class MNIST_DFT(Dataset):
    
    def __init__(self, rootpath, subpath=None):
        super().__init__()
        self.rootpath = rootpath
        self.subpath = subpath
        self.path = os.path.join(self.rootpath, self.subpath)
        self.img_path_list = os.listdir(self.path)
        
    def __getitem__(self, index):
        img_name = self.img_path_list[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        img_tensor = transform(img)[0].unsqueeze(0)
        label = 0
        return img_tensor, label
    
    def __len__(self):
        return len(self.img_path_list)
    
train_dataset = MNIST_DFT("../data/MNIST_DFT", "train")
test_set = MNIST_DFT("../data/MNIST_DFT", "test")

# # Loading the training dataset. We need to split it into a training and validation part
# train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

# # Loading the test set
# test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

# # We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0)


# Network Definition

# In[71]:


class MaskedConvolution(nn.Module):

    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer('mask', mask[None,None])

    def forward(self, x):
        self.conv.weight.data *= self.mask # Ensures zero's at masked positions
        return self.conv(x)


# In[72]:


class VerticalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size//2,:] = 0

        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0,kernel_size//2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


# In[73]:


class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


# In[74]:


class PixelCNN(nn.Module):

    def __init__(self, c_in, c_hidden):
        super().__init__()
        # self.save_hyperparameters()

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=4),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden)
        ])
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in * 256, kernel_size=1, padding=0)

        # self.example_input_array = train_set[0][0][None]

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        x = (x.float() / 255.0) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1]//256, out.shape[2], out.shape[3])
        return out

    def calc_likelihood(self, x):
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x)
        nll = F.cross_entropy(pred, x, reduction='none')
        bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))
        return bpd.mean()

    @torch.no_grad()
    def sample(self, img_shape, img=None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.long).to(device) - 1
        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:,c,h,w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:,:,:h+1,:])
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
    #     return [optimizer], [scheduler]

    # def training_step(self, batch, batch_idx):
    #     loss = self.calc_likelihood(batch[0])
    #     self.log('train_bpd', loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     loss = self.calc_likelihood(batch[0])
    #     self.log('val_bpd', loss)

    # def test_step(self, batch, batch_idx):
    #     loss = self.calc_likelihood(batch[0])
    #     self.log('test_bpd', loss)


# Loss function

# In[75]:


class Loss_fn(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, imgs):
        return (F.cross_entropy(outputs, imgs, reduction='none')).mean()


# Image saving&demonstration

# In[76]:


def show_imgs(imgs):
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    nrow = min(num_imgs, 4)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128)
    imgs = imgs.clamp(min=0, max=255)
    np_imgs = imgs.cpu().numpy()
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.show()
    plt.close()


# Training

# In[77]:


PixelCNN_model = PixelCNN(c_in=1, c_hidden=64)
PixelCNN_model = PixelCNN_model.to(device)
optimizer = optim.Adam(PixelCNN_model.parameters(), lr=Learning_rate)
loss_fn = Loss_fn()
loss_fn = loss_fn.to(device)
start_epoch = -1


if not os.path.isdir(os.path.join(LOG_PATH, Train_tag)):
        os.mkdir(os.path.join(LOG_PATH, Train_tag))
writer = SummaryWriter(os.path.join(LOG_PATH, Train_tag))
if not os.path.isdir(os.path.join(CHECKPOINT_PATH, Train_tag)):
        os.mkdir(os.path.join(CHECKPOINT_PATH, Train_tag))

# if resume from previous checkpoints
if Resume_from_checkpoint:
    path_checkpoint = RESUME_PATH
    checkpoint = torch.load(path_checkpoint)  
    PixelCNN_model.load_state_dict(checkpoint['net']) 
    # optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch']
    print("checkpoint {} successfully resumed. starting from epoch {}".format(RESUME_PATH, start_epoch+1))

epoch = start_epoch
train_iter = 0
validation_iter = 0
best_mean_loss = 100.0
for epoch_iter in range(Max_epoch):
    epoch = epoch + 1
    mean_loss = 0.0
    
    # train
    for imgs, _ in tqdm(train_loader, desc="epoch {} training ".format(epoch), leave=False):
        imgs = imgs.to(device)
        outputs = PixelCNN_model(imgs)
        loss = loss_fn(outputs, imgs)
        writer.add_scalar("training_loss", loss.item(), train_iter)
        train_iter += 1 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # validation
    with torch.no_grad():
        total_loss = 0.0
        validation_img_num = 0
        for imgs, _ in tqdm(val_loader, desc="epoch {} validating ".format(epoch), leave=False):
            imgs = imgs.to(device)
            outputs = PixelCNN_model(imgs)
            loss = loss_fn(outputs, imgs)
            total_loss += loss.item()
            validation_img_num += 1
            writer.add_scalar("validation_loss", loss.item(), validation_iter) 
            validation_iter += 1
        mean_loss = total_loss / validation_img_num
        writer.add_scalar("validation_loss_mean", mean_loss, epoch) 
    
    # save checkpoint every 5 epochs
    if(epoch%5==0):
        checkpoint = {
            "net": PixelCNN_model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch": epoch 
        }
        torch.save(checkpoint, os.path.join(os.path.join(CHECKPOINT_PATH, Train_tag), "epoch_{}.ckpt".format(epoch)))
        print("epoch_{}.ckpt saved!".format(epoch))
        samples = PixelCNN_model.sample(img_shape=(16,1,28,56)).cpu()
        writer.add_images("generated_DFT_sample", samples, epoch, dataformats="NCHW")
        restored_imgs = torch.zeros((samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3]//2))
        for i in range(restored_imgs.shape[0]):
            restored_imgs[i] = torch.tensor(inverse_DFT_transform(samples[i]))
        writer.add_images("restored_imgs", restored_imgs, epoch, dataformats="NCHW")
        
    # save the best model
    if(mean_loss < best_mean_loss):
        best_mean_loss = mean_loss
        checkpoint = {
            "net": PixelCNN_model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch": epoch 
        }
        torch.save(checkpoint, os.path.join(os.path.join(CHECKPOINT_PATH, Train_tag), "best_epoch.ckpt"))
        print("best_epoch_{}_loss_{}.ckpt".format(epoch, mean_loss))
        samples = PixelCNN_model.sample(img_shape=(16,1,28,56)).cpu()
        restored_imgs = torch.zeros((samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3]//2))
        for i in range(restored_imgs.shape[0]):
            restored_imgs[i] = torch.tensor(inverse_DFT_transform(samples[i]))
        writer.add_images("restored_imgs_for_best_model", restored_imgs, epoch, dataformats="NCHW")

# save the last model
if(epoch%5!=0):
    checkpoint = {
        "net": PixelCNN_model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "epoch": epoch 
    }
    torch.save(checkpoint, os.path.join(os.path.join(CHECKPOINT_PATH, Train_tag), "epoch_{}.ckpt".format(epoch)))
    print("epoch_{}.ckpt saved!".format(epoch))
    samples = PixelCNN_model.sample(img_shape=(16,1,28,56)).cpu()
    writer.add_images("generated_DFT_sample", samples, epoch, dataformats="NCHW")
    restored_imgs = torch.zeros((samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3]//2))
    for i in range(restored_imgs.shape[0]):
        restored_imgs[i] = torch.tensor(inverse_DFT_transform(samples[i]))
    writer.add_images("restored_imgs", restored_imgs, epoch, dataformats="NCHW")
    
    

