import medmnist
from medmnist import BloodMNIST
from medmnist import BreastMNIST
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
from torch.utils.data import DataLoader

def prep_dataA():
    print("\nPreparing Data for Task A...")
    # Define transformations (convert to PyTorch tensors)
    transform = Compose([ToTensor()])  # Normalizes pixel values to [0, 1]
    train_transform = Compose([
        RandomHorizontalFlip(),  # Randomly flip images horizontally
        RandomRotation(20),      # Randomly rotate images within a 20-degree range
        RandomResizedCrop(28, scale=(0.8, 1.0)),
        ToTensor(),              # Convert images to PyTorch tensors
    ])

    # Load datasets
    train_dataset = BreastMNIST(split='train', transform=train_transform, download=True)
    val_dataset = BreastMNIST(split='val', transform=transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=transform, download=True)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Data Prepared\n")

    return train_loader, val_loader, test_loader


def prep_dataB():
    print("\nPreparing Data for Task B...")
    # Define transformations (convert to PyTorch tensors)
    transform = Compose([ToTensor()])  # Normalizes pixel values to [0, 1]
    train_transform = Compose([
        RandomHorizontalFlip(),  # Randomly flip images horizontally
        RandomRotation(20),      # Randomly rotate images within a 20-degree range
        RandomResizedCrop(28, scale=(0.8, 1.0)),
        ToTensor(),              # Convert images to PyTorch tensors
    ])

    # Load datasets
    train_dataset = BloodMNIST(split='train', transform=train_transform, download=True)
    val_dataset = BloodMNIST(split='val', transform=transform, download=True)
    test_dataset = BloodMNIST(split='test', transform=transform, download=True)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Data Prepared\n")

    return train_loader, val_loader, test_loader

