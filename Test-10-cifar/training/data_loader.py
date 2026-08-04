import torch
import torchvision
from torchvision.transforms import v2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(128),                       
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
def get_data_loaders(batch_size=4, data_dir='./data', only_test=False):
    """Returns train and test dataloaders for CIFAR10.

    Set only_test=True to skip building the training set/loader entirely
    (e.g. for the attack scripts, which only ever use testloader). In that
    case trainloader is returned as None.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    g = torch.Generator()
    g.manual_seed(42)
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(128),                       
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True, generator=g
    )

    if only_test:
        return None, testloader

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5), 
        v2.RandomCrop(32, padding=4),
        v2.Resize(128),                       
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True 
    )

    return trainloader, testloader

def get_classes():
    """Returns CIFAR10 class names"""
    return ('airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')
