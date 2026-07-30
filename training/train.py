import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from training.model import Net, freeze_backbone, unfreeze_all
from timm.data import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.2, cutmix_alpha=0.0,
    num_classes=100, label_smoothing=0.1
)

def train_model(trainloader, device, epochs=90, accum_steps=8, arch='resnet50'):
    net = Net(num_classes=100, arch=arch)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    print("Stage 1: Training FC layer only (5 epochs)...")
    freeze_backbone(net)

    optimizer = optim.SGD(net.parameters(), lr=0.025, momentum=0.875, weight_decay=3.0517578125e-05)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    for epoch in range(5):
        net.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = mixup_fn(inputs, labels)

            with autocast('cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, labels) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            if i % 100 == 99:
                print(f'[Stage1-{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        scheduler.step()

    print("\nStage 2: Fine-tuning entire model (85 epochs)...")
    unfreeze_all(net)
    torch.cuda.empty_cache()

    optimizer = optim.SGD(net.parameters(), lr=0.025, momentum=0.875, weight_decay=3.0517578125e-05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 5)

    for epoch in range(5, epochs):
        net.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = mixup_fn(inputs, labels)

            with autocast('cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, labels) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            if i % 100 == 99:
                print(f'[Stage2-{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        scheduler.step()

    print('Finished Training')
    return net

def save_model(net, arch):
    path = f'./{arch}'
    torch.save(net.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model_class, path, device, arch):
    net = model_class(arch=arch)
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load(path, weights_only=True))
    return net