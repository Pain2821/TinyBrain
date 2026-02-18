"""
train_sequential.py - Train tasks one by one interactively
"""

import torch
from pathlib import Path
from core.bitbrain import BitBrain
from core.config import *
from utils.data import get_task_dataloaders
from training.train_task import train_task

def train_sequential():
    # Check if BitBrain already exists
    checkpoint_path = CHECKPOINT_DIR / 'bitbrain_sequential.pt'
    
    if checkpoint_path.exists():
        print("Found existing BitBrain!")
        checkpoint = torch.load(checkpoint_path)
        
        # Get info from checkpoint
        num_classes = checkpoint.get('current_num_classes', 2)
        task_history = checkpoint.get('task_history', [])
        
        print(f"  Previous tasks: {task_history}")
        print(f"  Pathways: {checkpoint.get('num_pathways', 0)}")
        
        # Initialize and load
        bitbrain = BitBrain(num_classes, device=DEVICE)
        bitbrain.load_state_dict(checkpoint['model_state'])
        
        print("  Loaded existing model!\n")
    else:
        print("No existing BitBrain found. Starting fresh.\n")
        bitbrain = None
    
    # Ask user what to train on
    print("="*70)
    print("What task do you want to train on?")
    print("="*70)
    
    # Show available datasets
    data_dir = Path('data')
    available = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(available):
        print(f"  {i+1}. {dataset}")
    
    print(f"  {len(available)+1}. Exit")
    
    choice = input("\nEnter number: ")
    
    try:
        choice = int(choice) - 1
        
        if choice == len(available):
            print("Exiting...")
            return
        
        if choice < 0 or choice >= len(available):
            print("Invalid choice!")
            return
        
        task_name = available[choice]
        data_path = data_dir / task_name
        
    except:
        print("Invalid input!")
        return
    
    # Load data
    print(f"\nLoading {task_name}...")
    train_loader, val_loader = get_task_dataloaders(str(data_path))
    
    num_classes = len(train_loader.dataset.classes) # type: ignore
    print(f"  Classes: {num_classes}")
    print(f"  Train samples: {len(train_loader.dataset)}") # type: ignore
    print(f"  Val samples: {len(val_loader.dataset)}\n") # type: ignore
    
    # Initialize BitBrain if first task
    if bitbrain is None:
        bitbrain = BitBrain(num_classes, device=DEVICE)
    else:
        # Reset for new task
        bitbrain.reset_fast_learner(
            new_num_classes=num_classes,
            keep_backbone=True
        )
    
    # Train
    print("="*70)
    print(f"Training on: {task_name}")
    print("="*70 + "\n")
    
    best_acc, history = train_task(
        bitbrain,
        train_loader,
        val_loader,
        task_name,
        epochs=EPOCHS_PER_TASK,
        device=DEVICE
    )
    
    # Consolidate
    print("\n[CONSOLIDATION]")
    consolidate = input("Consolidate this task into pathway? (y/n): ")
    
    if consolidate.lower() == 'y':
        bitbrain.consolidate_task(task_name)
        print("[OK] Task consolidated!\n")
    else:
        print("Skipped consolidation.\n")
    
    # Save
    torch.save({
        'model_state': bitbrain.state_dict(),
        'task_history': bitbrain.task_history,
        'num_pathways': len(bitbrain.pathways),
        'current_num_classes': num_classes
    }, checkpoint_path)
    
    print(f"Model saved to: {checkpoint_path}")
    
    # Continue?
    print("\n" + "="*70)
    again = input("Train another task? (y/n): ")
    
    if again.lower() == 'y':
        train_sequential()  # Recursive call
    else:
        print("\nDone! You can resume later by running this script again.")
        print(f"Current pathways: {len(bitbrain.pathways)}")
        print(f"Tasks trained: {bitbrain.task_history}")

if __name__ == "__main__":
    train_sequential()
