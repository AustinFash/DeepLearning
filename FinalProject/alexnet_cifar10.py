import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader

def get_device():
    # Choose GPU if available, otherwise CPU or MPS
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, use_dropout=True, use_batchnorm=True):
        super().__init__()
        # Feature extractor: five conv blocks with optional BatchNorm
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Classifier head: optional Dropout + three Linear layers
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())
        layers += [nn.Linear(256 * 4 * 4, 4096), nn.ReLU(inplace=True)]
        if use_dropout:
            layers.append(nn.Dropout())
        layers += [nn.Linear(4096, 4096), nn.ReLU(inplace=True)]
        layers += [nn.Linear(4096, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through features then classifier
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def run_single(config, device):
    # Unpack hyperparameters
    bs       = config['batch_size']
    lr       = config['lr']
    opt_name = config['optimizer']
    use_do   = config['use_dropout']
    use_bn   = config['use_batchnorm']
    epochs   = 5

    # Prepare data transforms
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010)),
    ])

    # Load CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(
        './data', train=True,  download=True, transform=tfm_train)
    testset  = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=tfm_test)

    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=(device.type=='cuda'))
    testloader  = DataLoader(
        testset,  batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=(device.type=='cuda'))

    # Instantiate model, loss, and optimizer
    model     = AlexNet(use_dropout=use_do, use_batchnorm=use_bn).to(device)
    criterion = nn.CrossEntropyLoss()
    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Training {opt_name}, lr={lr}, bs={bs}, drop={use_do}, bn={use_bn}")
    # Training loop
    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            # Accumulate statistics
            running_loss += loss.item() * labels.size(0)
            correct      += (outs.argmax(1) == labels).sum().item()
            total        += labels.size(0)
        # Print training stats for this epoch
        print(f"    → Epoch {epoch}/{epochs}  "
              f"Loss: {running_loss/total:.4f}, "
              f"Acc: {correct/total:.4f}, "
              f"Time: {time.time()-t0:.1f}s")

    # Final test evaluation
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            loss_sum += criterion(outs, labels).item() * labels.size(0)
            correct  += (outs.argmax(dim=1) == labels).sum().item()
            total    += labels.size(0)

    # Return summary of this run
    return {
        'optimizer':     opt_name,
        'lr':            lr,
        'batch_size':    bs,
        'use_dropout':   use_do,
        'use_batchnorm': use_bn,
        'test_loss':     loss_sum/total,
        'test_acc':      correct/total,
    }

def main():
    # Determine compute device
    device = get_device()
    print("Running sweep on", device)

    # Hyperparameter grid
    grid = {
        'optimizer':     ['SGD','Adam','RMSprop'],
        'lr':            [1e-2,1e-3,1e-4],
        'batch_size':    [32,64,128],
        'use_dropout':   [False, True],
        'use_batchnorm': [False, True],
    }
    results = []

    # Iterate over all 108 combinations
    for vals in itertools.product(*[grid[k] for k in grid]):
        config = dict(zip(grid.keys(), vals))
        print("Config:", config)
        res = run_single(config, device)
        print(f"  → Acc: {res['test_acc']:.3f}")
        results.append(res)

    # Place results into a DataFrame
    df = pd.DataFrame(results)

    # Print summaries
    print("\nTop 5 configurations by test accuracy:")
    print(df.sort_values('test_acc', ascending=False).head())

    max_acc = df['test_acc'].max()
    best_cfg = df.loc[df['test_acc'].idxmax()]
    print(f"\n→ Highest test accuracy achieved: {max_acc:.4f}")
    print("  Best config:", best_cfg.to_dict())

    print("\nAverage accuracy by optimizer & learning rate:")
    print(df.pivot_table(index='lr', columns='optimizer', values='test_acc'))

    print("\nAverage accuracy by batch size:")
    print(df.groupby('batch_size')['test_acc'].mean())

    print("\nAverage accuracy by Dropout/BatchNorm:")
    print(df.groupby(['use_dropout','use_batchnorm'])['test_acc'].mean())

if __name__ == '__main__':
    main()
