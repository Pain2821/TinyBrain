# core/utils_data.py
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from core.config import IMG_SIZE, NUM_WORKERS
from core.utils_logging import vlog
from typing import Tuple

def get_dataloaders(data_dir: str, img_size=IMG_SIZE, batch_size=32, num_workers=NUM_WORKERS) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    vlog(True, f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val samples")
    vlog(True, f"Classes: {train_ds.classes}")

    return train_loader, val_loader
