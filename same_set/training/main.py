from torch import device

from training.data_loader import get_data_loaders, get_classes
from training.model import Net
from training.train import train_model, save_model, load_model
from training.evaluate import evaluate_batch, evaluate_full, evaluate_per_class
from training.utils import get_device, imshow
import torchvision

def main():
    arch = 'densenet121' 
    device = get_device()
    classes = get_classes()
    batch_size = 32
    model_path = f'./{arch}.pt'
    
    # Load data
    print("Loading data...")
    trainloader, testloader = get_data_loaders(batch_size)
    
    # Visualize training data
    print("Visualizing training data...")
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))
    
    # Train
    print("Training model...")
    net = train_model(trainloader, device, epochs=90, accum_steps=8, arch=arch)
    save_model(net, model_path)
    
    # Evaluate on test batch
    print("\nEvaluating on test batch...")
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    
    net = load_model(Net, model_path, device, arch=arch)
    evaluate_batch(net, images, labels, device, classes)
    
    # Full evaluation
    print("\nFull evaluation...")
    evaluate_full(net, testloader, device)
    
    # Per-class evaluation
    evaluate_per_class(net, testloader, device, classes)

if __name__ == '__main__':
    main()
