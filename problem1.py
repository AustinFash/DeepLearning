import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# helper function to initialize weights using Xavier initialization
def init_cnn(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)


# baseline LeNet model 
lenet_baseline = nn.Sequential(
    # first convolutional block
    nn.Conv2d(3, 6, kernel_size=5, padding=2),  # (3,32,32) -> (6,32,32)
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),        # (6,32,32) -> (6,16,16)
    
    # second convolutional block
    nn.Conv2d(6, 16, kernel_size=5),              # (6,16,16) -> (16,12,12)
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),        # (16,12,12) -> (16,6,6)
    
    # dense block
    nn.Flatten(),                                # Flatten to (16*6*6 = 576)
    nn.Linear(16 * 6 * 6, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# dropout variant
lenet_dropout = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Flatten(),
    nn.Linear(16 * 6 * 6, 120),
    nn.Sigmoid(),
    nn.Dropout(p=0.5),    # dropout inserted here
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# batch normalization variant
lenet_batchnorm = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=2),
    nn.BatchNorm2d(6),    # Batch normalization applied here
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Flatten(),
    nn.Linear(16 * 6 * 6, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# initialize weights for all models
lenet_baseline.apply(init_cnn)
lenet_dropout.apply(init_cnn)
lenet_batchnorm.apply(init_cnn)

# training and evaluation function
def train_and_evaluate(model, trainloader, testloader, num_epochs=20, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    epoch_logs = []  # to store each epoch's log
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()          # eeset gradients
            outputs = model(inputs)        # forward pass
            loss = criterion(outputs, labels)
            loss.backward()                # backpropagation
            optimizer.step()               # update weights
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        
        # evaluation on test data
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100.0 * correct / total
        
        log_str = f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Test Accuracy: {test_accuracy:.2f}%"
        print(log_str)
        epoch_logs.append(log_str)
    
    total_time = time.time() - start_time
    print("Total training time: {:.2f} seconds".format(total_time))
    return model, epoch_logs

#  main function
if __name__ == "__main__":
    # use MPS on a Mac with an M1 chip
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)
    num_workers_val = 0 if device.type == "mps" else 2

    # data transformations: convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # download and prepare the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=num_workers_val)

    # download and prepare the CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers_val)

    num_epochs = 20  # number of epochs for training
    
    # train and evaluate the Baseline LeNett
    print("\nTraining Baseline LeNet Model")
    model_baseline, logs_baseline = train_and_evaluate(lenet_baseline, trainloader, testloader, num_epochs, device)
    
    # train and evaluate the LeNet model with Dropoutt
    print("\nTraining LeNet Model with Dropout")
    model_dropout, logs_dropout = train_and_evaluate(lenet_dropout, trainloader, testloader, num_epochs, device)
    
    # train and evaluate the LeNet model with Batch Normalizationt
    print("\nTraining LeNet Model with Batch Normalization")
    model_batchnorm, logs_batchnorm = train_and_evaluate(lenet_batchnorm, trainloader, testloader, num_epochs, device)
