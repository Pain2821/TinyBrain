# core/utils_quant.py
"""
Simple helpers for centroid quantization and int8 blob representation.
"""
import numpy as np
from core.config import INT8_MAX

def quantize_centroid_int8(centroid: np.ndarray):
    centroid = centroid.astype(np.float32)
    max_abs = float(np.max(np.abs(centroid)) + 1e-9)
    scale = max_abs / INT8_MAX
    quantized = np.round(centroid / scale).astype(np.int8)
    return scale, quantized
