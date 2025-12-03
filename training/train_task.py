import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time

from core.bitbrain import BitBrain
from core.config import *


def train_task(
    bitbrain: BitBrain,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_name: str,
    epochs: int = EPOCHS_PER_TASK,
    lr: float = LEARNING_RATE,
    device: str = DEVICE,
    verbose: bool = VERBOSE
):
    print(f"\n{'='*70}")
    print(f"Training Task: {task_name}")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}, LR: {lr}, Device: {device}")
    print(f"Fast Learner: {bitbrain._get_model_size_mb():.1f} MB")
    print(f"Existing pathways: {len(bitbrain.pathways)}")
    print(f"{'='*70}\n")
    
    # Optimizer - only Fast Learner and Router are trainable
    optimizer = torch.optim.AdamW([
        {'params': bitbrain.fast_learner.parameters(), 'lr': lr},
        {'params': bitbrain.router.parameters(), 'lr': lr * ROUTER_LR_SCALE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ============ TRAINING ============
        bitbrain.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward
            output = bitbrain(x)
            loss = criterion(output, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
            # Print progress
            if verbose and batch_idx % PRINT_EVERY == 0:
                batch_acc = (preds == y).float().mean().item()
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # ============ VALIDATION ============
        bitbrain.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = bitbrain(x)
                preds = output.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc:  {train_acc:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"  ⭐ New best validation accuracy!")
            
            # Save checkpoint
            checkpoint_path = CHECKPOINT_DIR / f'bitbrain_{task_name}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state': bitbrain.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'task_name': task_name
            }, checkpoint_path)
        
        print()
    
    print(f"{'='*70}")
    print(f"Task '{task_name}' training complete!")
    print(f"  Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"{'='*70}\n")
    
    return best_val_acc, history