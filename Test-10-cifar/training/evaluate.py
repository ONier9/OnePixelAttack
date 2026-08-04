import torch
from training.model import Net

def evaluate_batch(net, images, labels, device, classes):
    """Evaluate a single batch and print predictions"""
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = net(images)
    
    _, predicted = torch.max(outputs, 1)
    
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))
    print('Predicted:  ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(predicted))))

def evaluate_full(net, testloader, device):
    """Evaluate accuracy on entire test set"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
    return accuracy

def evaluate_per_class(net, testloader, device, classes):
    """Evaluate accuracy for each class"""
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    print('\nAccuracy per class:')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'  {classname:5s}: {accuracy:.1f} %')