BitBrain is a proposed neural network architecture designed to solve the 'Efficiency-Plasticity Dilemma' in Artificial Intelligence.

 Z:/Personal/TinyBrain/venv/Scripts/Activate.ps1

 venv/Scripts/Activate

# Normal training with quantization
python main.py --mode train

# Training WITHOUT quantization (to test if it helps)
python main.py --mode train --no-quantize

# Test quantization quality
python main.py --mode test_quant

# Compare external vs internal
python main.py --mode compare
