import torch
from pathlib import Path
import json
import random
import numpy as np

from core.bitbrain import BitBrain
from core.config import *
from utils.data import get_task_dataloaders
from training.continual_training import continual_learning
from test import test_all_tasks


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_task_configs(tasks_file: str = ""):
    if tasks_file:
        with open(tasks_file, 'r', encoding='utf-8') as f:
            task_configs = json.load(f)
        if not isinstance(task_configs, list):
            raise ValueError("Task config file must contain a JSON array.")
        return task_configs

    return [
        {
            'name': 'cats_dogs',
            'data_dir': 'data',
            'num_classes': 2,
            'epochs': EPOCHS_PER_TASK
        }
    ]


def main(tasks_file: str = "", epochs_override: int = 0, quantize_pathways: bool = QUANTIZE_PATHWAYS):
    """
    Main experiment: Train BitBrain on sequence of tasks
    """
    
    print("\n" + "="*70)
    print("BitBrain: Internal Memory Continual Learning")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Quantization enabled: {quantize_pathways}")
    print(f"Threshold percentile: {THRESHOLD_PERCENTILE}")
    print(f"Top-K routing: {ENABLE_TOPK_ROUTING} (k={TOPK_PATHWAYS})")
    print(f"AMP enabled: {USE_AMP and DEVICE.startswith('cuda')}")
    print("="*70 + "\n")
    
    task_configs = load_task_configs(tasks_file)
    
    # ========== INITIALIZE BITBRAIN ==========
    
    # Start with num_classes of first task
    initial_num_classes = task_configs[0]['num_classes']
    
    bitbrain = BitBrain(
        num_classes=initial_num_classes,
        base_arch=BASE_ARCH,
        device=DEVICE
    )
    
    print(f"BitBrain initialized with {initial_num_classes} classes\n")
    
    # ========== LOAD DATA FOR ALL TASKS ==========
    
    print("Loading datasets...")
    
    task_data = []
    for task_config in task_configs:
        data_dir = task_config['data_dir']
        if epochs_override > 0:
            task_config['epochs'] = epochs_override
        
        # Check if data exists
        if not Path(data_dir).exists():
            print(f"WARNING: Data directory not found: {data_dir}")
            print(f"         Skipping task: {task_config['name']}")
            continue
        
        # Load dataloaders
        train_loader, val_loader = get_task_dataloaders(
            data_dir,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        
        # Add to config
        task_config['train_loader'] = train_loader
        task_config['val_loader'] = val_loader
        task_data.append(task_config)
        
        print(f"  [OK] {task_config['name']}: "
              f"{len(train_loader.dataset)} train, " # type: ignore
              f"{len(val_loader.dataset)} val") # type: ignore
    
    if len(task_data) == 0:
        print("\nERROR: No valid datasets found!")
        print("Please check your data paths in the configuration.")
        return
    
    print(f"\nTotal tasks to train: {len(task_data)}\n")
    
    # ========== CONTINUAL LEARNING ==========
    
    # Train on all tasks sequentially
    results = continual_learning(
        bitbrain,
        task_data,
        device=DEVICE,
        quantize_pathways=quantize_pathways
    )

    # ========== NEW: PATHWAY QUALITY CHECK ==========
    
    print("\n" + "="*70)
    print("Pathway Quality Check")
    print("="*70 + "\n")
    
    # Test each pathway independently
    from test import test_pathway_quality
    
    for task_config in task_data:
        test_loader = task_config['val_loader']
        test_pathway_quality(bitbrain, test_loader, device=DEVICE)
        break  # Just test on first task for now
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "="*70)
    print("Final Evaluation: Testing on ALL tasks")
    print("="*70 + "\n")
    
    # Prepare test loaders for all tasks
    test_loaders = {}
    for task_config in task_data:
        # Use validation set as test set
        test_loaders[task_config['name']] = task_config['val_loader']
    
    # Test on all tasks
    final_results = test_all_tasks(
        bitbrain,
        test_loaders,
        device=DEVICE
    )
    
    # ========== FORGETTING ANALYSIS ==========
    
    # To measure forgetting, you need accuracy after each task
    # This requires testing on all previous tasks after each new task
    # For simplicity, we show final results only
    
    print("\n" + "="*70)
    print("Final Results Summary")
    print("="*70)
    
    print(f"\nMemory Usage:")
    memory = bitbrain.get_memory_breakdown()
    print(f"  Fast Learner: {memory['fast_learner_mb']:.2f} MB")
    print(f"  Pathways: {memory['pathways_mb']:.2f} MB ({memory['num_pathways']} total)")
    print(f"  Router: {memory['router_mb']:.2f} MB")
    print(f"  Total: {memory['total_mb']:.2f} MB")
    
    print(f"\nAccuracy on Each Task:")
    for task_name, task_result in final_results.items():
        if task_name == 'average_accuracy':
            continue
        print(f"  {task_name:<20}: {task_result['accuracy']:.4f}")
    
    print(f"\nAverage Accuracy: {final_results['average_accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70 + "\n")
    
    # ========== SAVE MODEL ==========
    
    save_path = CHECKPOINT_DIR / 'bitbrain_final.pt'
    torch.save({
        'model_state': bitbrain.state_dict(),
        'task_history': bitbrain.task_history,
        'num_pathways': len(bitbrain.pathways),
        'final_results': final_results
    }, save_path)
    
    print(f"Model saved to: {save_path}\n")


def test_quantization_quality():
    """
    Quick test: Compare quantized vs non-quantized pathways
    
    This tests if ternary quantization hurts accuracy too much
    """
    print("\n" + "="*70)
    print("Testing Quantization Quality")
    print("="*70 + "\n")
    
    from core.quantization import quantize_to_ternary, estimate_ternary_size
    
    # Create a simple test network
    test_model = torch.nn.Sequential(
        torch.nn.Linear(1280, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10)
    )
    
    # Get original size
    original_size = estimate_ternary_size(test_model)
    print(f"Original model:")
    print(f"  FP32: {original_size['fp32_mb']:.2f} MB")
    print(f"  Params: {original_size['params']:,}")
    
    # Quantize
    print(f"\nQuantizing to ternary...")
    quantized_model, stats = quantize_to_ternary(test_model, threshold_percentile=0.7) # type: ignore
    
    # Get quantized size
    quantized_size = estimate_ternary_size(quantized_model)
    print(f"\nQuantized model:")
    print(f"  Ternary: {quantized_size['ternary_mb']:.2f} MB")
    print(f"  Compression: {original_size['fp32_mb'] / quantized_size['ternary_mb']:.1f}x")
    print(f"  Sparsity: {(1 - stats['non_zero_params']/stats['total_params'])*100:.1f}%")
    
    print("\n" + "="*70 + "\n")


def compare_external_vs_internal():
    """
    Comparison: External memory (your old code) vs Internal memory (new code)
    
    Shows the key differences
    """
    print("\n" + "="*70)
    print("External vs Internal Memory Comparison")
    print("="*70 + "\n")
    
    print("EXTERNAL MEMORY (Old Code):")
    print("  - Pathways: Activation centroids saved to disk")
    print("  - Storage: JSON files + .int8 blobs")
    print("  - Memory/pathway: ~0.25 MB")
    print("  - Routing: Load from disk, compare similarities")
    print("  - Quantization: INT8 (8 bits)")
    print("  - Advantages: Simple, unlimited capacity")
    print("  - Disadvantages: Not truly 'in the brain', slower\n")
    
    print("INTERNAL MEMORY (New Code):")
    print("  - Pathways: Complete frozen networks")
    print("  - Storage: nn.ModuleList in model")
    print("  - Memory/pathway: ~0.875 MB (ternary)")
    print("  - Routing: Integrated gating network")
    print("  - Quantization: Ternary 1.58-bit")
    print("  - Advantages: True synaptic pathways, faster routing")
    print("  - Disadvantages: More complex, higher memory\n")
    
    print("KEY DIFFERENCE:")
    print("  External: Stores REPRESENTATIONS (activations)")
    print("  Internal: Stores COMPUTATION (network weights)")
    print("  Internal is what the PDF describes!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BitBrain Continual Learning')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test_quant', 'compare'],
                       help='Mode to run')
    parser.add_argument('--tasks-file', type=str, default='',
                       help='JSON file with task configs (list of task objects)')
    parser.add_argument('--epochs', type=int, default=0,
                       help='Override epochs for all tasks (0 keeps per-task value)')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-quantize', action='store_true',
                       help='Disable ternary quantization (test only)')
    
    args = parser.parse_args()
    
    # Override config if requested
    quantize_pathways = not args.no_quantize
    if not quantize_pathways:
        print("[WARNING] Quantization disabled - pathways will be FP32\n")
    
    # Run selected mode
    if args.mode == 'train':
        set_seed(args.seed)
        main(
            tasks_file=args.tasks_file,
            epochs_override=args.epochs,
            quantize_pathways=quantize_pathways
        )
    elif args.mode == 'test_quant':
        test_quantization_quality()
    elif args.mode == 'compare':
        compare_external_vs_internal()
