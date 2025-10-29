import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_cifar10_loaders(batch_size=128, num_workers=2):
    """Pravi train i test DataLoader za CIFAR-10 dataset."""
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616)),
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )

@torch.no_grad()
def evaluate(model, loader, device):
    """Računa tačnost modela na datom skupu."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total
