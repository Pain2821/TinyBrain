import os
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from core.config import *


def get_transforms(img_size: int = IMG_SIZE, augment: bool = True):

    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return transform


def get_dataloader(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    split: str = 'train',
    num_workers: int = NUM_WORKERS,
    shuffle: bool = None # type: ignore
):

    augment = (split == 'train')
    if shuffle is None:
        shuffle = (split == 'train')
    
    transform = get_transforms(augment=augment)
    
    dataset_path = Path(data_dir) / split
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    return dataloader


def get_task_dataloaders(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
):

    train_loader = get_dataloader(
        data_dir,
        batch_size=batch_size,
        split='train',
        num_workers=num_workers
    )
    
    val_loader = get_dataloader(
        data_dir,
        batch_size=batch_size,
        split='val',
        num_workers=num_workers
    )
    
    return train_loader, val_loader