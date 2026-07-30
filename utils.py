import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img):
    """Display an image tensor with ImageNet normalization"""
    img = img.clone()
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    for i in range(3):
        img[i] = img[i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]
    
    img = torch.clamp(img, 0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_device():
    """Get the appropriate device (CUDA or CPU)"""
    if torch.accelerator.is_available():
        device = torch.device(torch.accelerator.current_accelerator().type)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device