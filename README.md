BitBrain is a continual-learning architecture with:
- A plastic Fast Learner for new tasks
- Frozen pathway memory for old tasks
- Router-based pathway selection with Top-K sparse execution

## Setup (Windows PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Layout
Use this simple structure (one folder per task):
```text
datasets/
  task_1/
    train/
      class_a/
      class_b/
    val/
      class_a/
      class_b/
  task_2/
    train/
      class_x/
      class_y/
    val/
      class_x/
      class_y/
```

Pre-created for you:
- `datasets/cats_dogs` (already filled)
- `datasets/fruits` (empty template: add apple/banana images)

## Training
Default single-task run (uses `data/`):
```powershell
python main.py --mode train
```

Use a task config file:
```powershell
python main.py --mode train --tasks-file training/experiments/tasks_example.json
```

Continual-learning template (cats/dogs then fruits):
```powershell
python main.py --mode train --tasks-file training/experiments/tasks_continual_template.json
```

For a ready-made forgetting test (4 tasks, each 3 classes):
```powershell
python main.py --mode train --tasks-file training/experiments/tasks_forgetting_check.json --epochs 6 --seed 42 --no-quantize
```

Override epochs for all tasks and set seed:
```powershell
python main.py --mode train --tasks-file training/experiments/tasks_example.json --epochs 12 --seed 42
```

Disable pathway quantization:
```powershell
python main.py --mode train --no-quantize
```

## Utility Modes
```powershell
python main.py --mode test_quant
python main.py --mode compare
```

## Outputs
- Model checkpoint: `checkpoints_internal/bitbrain_final.pt`
- Continual-learning metrics: `logs_internal/continual_learning_results.json`
