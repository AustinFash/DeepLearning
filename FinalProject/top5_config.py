import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd

def get_device():
    # Select GPU if available, else fall back to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, use_dropout=True, use_batchnorm=True):
        super().__init__()
        # Feature extractor: 5 convolutional blocks with optional BatchNorm
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classifier: up to two Dropout layers and three Linear layers
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())              
        layers += [nn.Linear(256 * 4 * 4, 4096), nn.ReLU(inplace=True)]
        if use_dropout:
            layers.append(nn.Dropout())              
        layers += [nn.Linear(4096, 4096), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(4096, num_classes)) 
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through features and classifier
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten to (batch, features)
        return self.classifier(x)

def evaluate_model(cfg, device, trainloader, testloader, max_epochs=30):
    bs, opt_name, lr = cfg['batch_size'], cfg['optimizer'], cfg['lr']
    use_do, use_bn = cfg['use_dropout'], cfg['use_batchnorm']

    # Initialize model and move to device
    model = AlexNet(use_dropout=use_do, use_batchnorm=use_bn).to(device)
    criterion = nn.CrossEntropyLoss()  # Classification loss

    # Choose optimizer
    optimizer = {
        'SGD':    optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'RMSprop':optim.RMSprop(model.parameters(), lr=lr),
        'Adam':   optim.Adam(model.parameters(), lr=lr),
    }[opt_name]

    # Determine which epochs to test on (every 5 epochs)
    test_epochs = set(range(5, max_epochs+1, 5))
    test_accs = {}

    # Training loop
    for epoch in range(1, max_epochs+1):
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0
        model.train()

        # Iterate over training batches
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss and correct predictions
            running_loss += loss.item() * labels.size(0)
            correct     += (outputs.argmax(1) == labels).sum().item()
            total       += labels.size(0)

        # Compute epoch statistics
        train_loss = running_loss / total
        train_acc  = correct / total
        epoch_time = time.time() - start_time

        # Log message
        msg = (f"Config {opt_name} lr={lr} bs={bs} Epoch {epoch:2d}/{max_epochs} | "
               f"Train L: {train_loss:.4f}, A: {train_acc:.4f} | Time: {epoch_time:.1f}s")

        # Perform test evaluation at designated epochs
        if epoch in test_epochs:
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for imgs, labels in testloader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    test_correct += (outputs.argmax(1) == labels).sum().item()
                    test_total   += labels.size(0)
            test_acc = test_correct / test_total
            test_accs[f'acc_epoch_{epoch}'] = test_acc
            msg += f" | Test Acc: {test_acc:.4f}"

        print(msg)

    # Return config plus all recorded test accuracies
    return {**cfg, **test_accs}

if __name__ == '__main__':
    # Define the five best configurations to evaluate from Phase I
    configs = [
        {'optimizer':'Adam','lr':1e-4,'batch_size':32,  'use_dropout':False,'use_batchnorm':True},
        {'optimizer':'SGD', 'lr':1e-2,'batch_size':64,  'use_dropout':False,'use_batchnorm':True},
        {'optimizer':'SGD', 'lr':1e-2,'batch_size':128, 'use_dropout':False,'use_batchnorm':True},
        {'optimizer':'SGD', 'lr':1e-3,'batch_size':32,  'use_dropout':False,'use_batchnorm':True},
        {'optimizer':'Adam','lr':1e-4,'batch_size':64,  'use_dropout':False,'use_batchnorm':True},
    ]

    device = get_device()
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True  #

    # Data augmentation and normalization for training
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010)),
    ])
    # Only normalization for testing
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010)),
    ])

    # Load CIFAR-10 dataset
    data_train = torchvision.datasets.CIFAR10(
        './data', train=True, download=True, transform=tf_train)
    data_test  = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=tf_test)

    results = []
    # Evaluate each configuration in turn
    for cfg in configs:
        trainloader = DataLoader(
            data_train, batch_size=cfg['batch_size'],
            shuffle=True,  num_workers=4, pin_memory=True,
            persistent_workers=True)
        testloader  = DataLoader(
            data_test,  batch_size=cfg['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True,
            persistent_workers=True)

        print("\nEvaluating config:", cfg)
        res = evaluate_model(cfg, device, trainloader, testloader, max_epochs=30)
        results.append(res)

    # Summarize and save
    df = pd.DataFrame(results)
    print("\nFinal summary (test acc every 5 epochs up to 30):")
    print(df)
    df.to_csv('top5_longrun_summary.csv', index=False)
    print("Saved results to top5_longrun_summary.csv")
