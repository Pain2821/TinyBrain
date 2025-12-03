import torch
from torch.utils.data import DataLoader
from typing import List, Dict
import json
from pathlib import Path

from core.bitbrain import BitBrain
from core.config import *
from training.train_task import train_task


def continual_learning(
    bitbrain: BitBrain,
    task_configs: List[Dict],
    device: str = DEVICE
):
    print("\n" + "="*70)
    print("BitBrain Continual Learning")
    print("="*70)
    print(f"Total tasks: {len(task_configs)}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    results = {
        'tasks': [],
        'memory_history': [],
        'routing_history': []
    }
    
    for task_idx, task_config in enumerate(task_configs):
        task_name = task_config['name']
        train_loader = task_config['train_loader']
        val_loader = task_config['val_loader']
        num_classes = task_config.get('num_classes', bitbrain.num_classes)
        epochs = task_config.get('epochs', EPOCHS_PER_TASK)
        
        print(f"\n{'#'*70}")
        print(f"# Task {task_idx + 1}/{len(task_configs)}: {task_name}")
        print(f"{'#'*70}\n")
        
        # Train on this task
        best_acc, history = train_task(
            bitbrain,
            train_loader,
            val_loader,
            task_name,
            epochs=epochs,
            device=device
        )
        
        # Record results
        task_result = {
            'name': task_name,
            'task_idx': task_idx,
            'best_val_acc': best_acc,
            'history': history
        }
        results['tasks'].append(task_result)
        
        # Consolidate this task into pathway
        print(f"\n[CONSOLIDATION PHASE]")
        bitbrain.consolidate_task(
            task_name,
            threshold_percentile=THRESHOLD_PERCENTILE,
            quantize=QUANTIZE_PATHWAYS
        )
        
        # Memory breakdown after consolidation
        memory = bitbrain.get_memory_breakdown()
        results['memory_history'].append(memory)
        
        print(f"\n[MEMORY STATUS]")
        print(f"  Fast Learner: {memory['fast_learner_mb']:.2f} MB")
        print(f"  Pathways: {memory['pathways_mb']:.2f} MB ({memory['num_pathways']} total)")
        print(f"  Router: {memory['router_mb']:.2f} MB")
        print(f"  Total: {memory['total_mb']:.2f} MB\n")
        
        # Routing analysis
        if len(bitbrain.pathways) > 0:
            routing_stats = bitbrain.analyze_routing(print_results=True)
            results['routing_history'].append(routing_stats)
        
        # Reset Fast Learner for next task (if not last task)
        if task_idx < len(task_configs) - 1:
            next_num_classes = task_configs[task_idx + 1].get(
                'num_classes', num_classes
            )
            
            print(f"[RESET FOR NEXT TASK]")
            bitbrain.reset_fast_learner(
                new_num_classes=next_num_classes,
                keep_backbone=KEEP_BACKBONE_ON_RESET
            )
    
    # Final summary
    print("\n" + "="*70)
    print("Continual Learning Complete!")
    print("="*70)
    print(f"Total tasks trained: {len(results['tasks'])}")
    print(f"Total pathways: {len(bitbrain.pathways)}")
    print(f"Total memory: {bitbrain.get_total_memory():.2f} MB")
    print("\nPer-task validation accuracy:")
    for task_result in results['tasks']:
        print(f"  {task_result['name']:<20}: {task_result['best_val_acc']:.4f}")
    print("="*70 + "\n")
    
    # Save results
    results_path = LOG_DIR / 'continual_learning_results.json'
    with open(results_path, 'w') as f:
        # Convert non-serializable objects
        save_results = {
            'tasks': [
                {
                    'name': t['name'],
                    'task_idx': t['task_idx'],
                    'best_val_acc': t['best_val_acc']
                }
                for t in results['tasks']
            ],
            'memory_history': [
                {k: v for k, v in m.items() if k != 'pathways'}
                for m in results['memory_history']
            ]
        }
        json.dump(save_results, f, indent=2)
    
    print(f"Results saved to: {results_path}\n")
    
    return results