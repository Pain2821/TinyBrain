# training/train_bitbrain.py
import os
from pathlib import Path
import torch
import torch.nn as nn
from core.utils_data import get_dataloaders
from core.fast_learner import MobileNetFastLearner
from core.memory_bank import MemoryBank
from core.router import Router
from core.consolidator import Consolidator
from core.config import TORCH_DEVICE, CHECKPOINT_DIR, LEARNING_RATE, EPOCHS, BATCH_SIZE
from core.utils_logging import log

def train_bitbrain(data_dir: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, checkpoint_dir: str = None): # type: ignore
    checkpoint_dir = checkpoint_dir or str(CHECKPOINT_DIR)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(train_loader.dataset.classes) # type: ignore
    class_names = train_loader.dataset.classes # type: ignore

    model = MobileNetFastLearner(num_classes=num_classes, pretrained=True, capture_stages=5).to(TORCH_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    memory_bank = MemoryBank(str(Path("memory_bank")))
    router = Router(memory_bank)
    consolidator = Consolidator(model, memory_bank, router)

    best_val_acc = 0.0
    for epoch in range(epochs):
        log("\n" + "="*60)
        log(f"Epoch {epoch+1}/{epochs}")
        log("="*60)
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(TORCH_DEVICE), y.to(TORCH_DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += x.size(0)
            sample_size = min(x.size(0), 32)
            consolidator.add_to_buffer(x[:sample_size].detach(), y[:sample_size].detach())
            if batch_idx % 20 == 0:
                batch_acc = (preds == y).float().mean().item()
                log(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, acc={batch_acc:.4f}")
        scheduler.step()
        train_loss = epoch_loss / epoch_total if epoch_total > 0 else 0.0
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        # simple evaluation
        def evaluate_local(model, dataloader):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xx, yy in dataloader:
                    xx, yy = xx.to(TORCH_DEVICE), yy.to(TORCH_DEVICE)
                    logits = model(xx)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yy).sum().item()
                    total += yy.size(0)
            return correct/total if total>0 else 0.0
        val_acc = evaluate_local(model, val_loader)
        log(f"\n  📊 Epoch {epoch+1} Results:")
        log(f"     Train Loss: {train_loss:.4f}")
        log(f"     Train Acc:  {train_acc:.4f}")
        log(f"     Val Acc:    {val_acc:.4f}")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_best.pt"))
            log("     ⭐ New best model saved!")
        if consolidator.should_consolidate():
            consolidator.run_consolidation(class_names)
    log(f"\nTraining Complete! Best validation accuracy: {best_val_acc:.4f}")
    return model, memory_bank, router

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train_bitbrain(args.data, epochs=args.epochs, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir)
