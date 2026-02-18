import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np

from core.bitbrain import BitBrain
from core.config import *


def test_on_task(
    bitbrain: BitBrain,
    test_loader: DataLoader,
    task_name: str,
    device: str = DEVICE,
    analyze_routing: bool = True
) -> Dict:
    bitbrain.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_gates = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if analyze_routing and len(bitbrain.pathways) > 0:
                output, routing_info = bitbrain(x, return_routing_info=True)
                if isinstance(routing_info, dict):
                    gates = routing_info.get('gates')
                    if isinstance(gates, torch.Tensor):
                        all_gates.append(gates)
            else:
                output = bitbrain(x)
            
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    accuracy = correct / total
    
    results = {
        'task_name': task_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    
    # Routing analysis
    if len(all_gates) > 0:
        all_gates = torch.cat(all_gates, dim=0)  # [N, num_pathways]
        avg_gates = all_gates.mean(dim=0)
        
        results['routing'] = {
            'avg_pathway_weights': avg_gates.numpy().tolist(),
            'most_used_pathway': int(avg_gates.argmax()),
            'pathway_usage': {
                i: float(avg_gates[i])
                for i in range(len(avg_gates))
            }
        }
    
    return results


def test_all_tasks(
    bitbrain: BitBrain,
    task_loaders: Dict[str, DataLoader],
    device: str = DEVICE
) -> Dict:

    print("\n" + "="*70)
    print("Testing on All Tasks")
    print("="*70 + "\n")
    
    results = {}
    
    for task_name, test_loader in task_loaders.items():
        print(f"Testing on: {task_name}")
        task_results = test_on_task(
            bitbrain,
            test_loader,
            task_name,
            device,
            analyze_routing=True
        )
        
        results[task_name] = task_results
        
        print(f"  Accuracy: {task_results['accuracy']:.4f}")
        
        if 'routing' in task_results:
            most_used = task_results['routing']['most_used_pathway']
            usage = task_results['routing']['avg_pathway_weights'][most_used]
            print(f"  Most used pathway: {most_used} ({usage*100:.1f}%)")
        
        print()
    
    # Summary
    print("="*70)
    print("Summary:")
    print(f"{'Task':<20} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
    print("-"*70)
    
    accuracies = []
    for task_name, task_result in results.items():
        acc = task_result['accuracy']
        correct = task_result['correct']
        total = task_result['total']
        accuracies.append(acc)
        print(f"{task_name:<20} {acc:>10.4f}   {correct:>8d}   {total:>8d}")
    
    avg_acc = np.mean(accuracies)
    print("-"*70)
    print(f"{'Average':<20} {avg_acc:>10.4f}")
    print("="*70 + "\n")
    
    results['average_accuracy'] = avg_acc
    
    return results


def measure_forgetting(
    task_accuracies: Dict[str, List[float]],
    task_order: List[str]
) -> Dict:
    forgetting = {}
    
    for i, task_name in enumerate(task_order[:-1]):  # Exclude last task
        accs = task_accuracies[task_name]
        
        # Accuracy right after training this task
        acc_initial = accs[i]
        
        # Accuracy after training all subsequent tasks
        acc_final = accs[-1]
        
        # Forgetting = drop in accuracy
        forget = acc_initial - acc_final
        
        forgetting[task_name] = {
            'initial_acc': acc_initial,
            'final_acc': acc_final,
            'forgetting': forget,
            'retention_rate': acc_final / acc_initial if acc_initial > 0 else 0
        }
    
    # Average forgetting
    avg_forgetting = np.mean([f['forgetting'] for f in forgetting.values()])
    avg_retention = np.mean([f['retention_rate'] for f in forgetting.values()])
    
    return {
        'per_task': forgetting,
        'avg_forgetting': avg_forgetting,
        'avg_retention_rate': avg_retention
    }

def test_pathway_quality(
    bitbrain: BitBrain,
    test_loader: DataLoader,
    device: str = 'cuda'
):
    """
    Test pathway quality independently
    
    This helps diagnose if pathways are broken
    """
    if len(bitbrain.pathways) == 0:
        print("No pathways to test")
        return
    
    print("\n" + "="*70)
    print("Pathway Quality Analysis")
    print("="*70 + "\n")
    
    bitbrain.eval()
    
    for pathway_idx, pathway in enumerate(bitbrain.pathways):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Test pathway independently
                pathway_pred = pathway(x)
                preds = pathway_pred.argmax(dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total
        print(f"Pathway {pathway_idx} ({pathway.task_name}):")
        print(f"  Independent accuracy: {accuracy:.4f}")
        print(f"  Memory: {pathway.memory_mb:.2f} MB")
        print(f"  Sparsity: {pathway.get_sparsity()*100:.1f}%") # type: ignore
        print()
    
    # Also test Fast Learner
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            features = bitbrain.extract_features(x)
            fast_pred = bitbrain.fast_learner.classifier(features) # type: ignore
            preds = fast_pred.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    print(f"Fast Learner:")
    print(f"  Independent accuracy: {correct/total:.4f}")
    print()
    
    print("="*70 + "\n")
    
if __name__ == "__main__":
    print("BitBrain Testing Module")
    print("This file contains testing utilities.")
    print("Import and use in your main training script.")
