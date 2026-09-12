"""
Offline / Edge Optimization via ONNX Runtime (Enhancement B1 & A3)
Exports PyTorch DR Classifier to ONNX format and executes high-speed CPU inference.
Zero-cloud dependencies, sub-50ms latency on rural laptops/tablets.
"""

import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from modules.ai_models import DRClassifierBackbone

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ONNX_PATH = os.path.join(MODEL_DIR, "dr_classifier.onnx")

import sys
import io

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def export_dr_model_to_onnx(output_path=ONNX_PATH, force=False):
    """
    Exports the PyTorch DRClassifierBackbone architecture to a self-contained ONNX file
    with opset version 18. Compatible with ONNX Runtime, TensorRT, and MATLAB importONNXNetwork.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If model exists and is valid, reuse it to avoid file lock collisions on Windows
    if os.path.exists(output_path) and not force:
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return output_path, file_size_mb
        except Exception:
            pass # Re-export if corrupt
    
    # Instantiate PyTorch model in evaluation mode
    model = DRClassifierBackbone(num_classes=5)
    model.eval()
    
    # Dummy input: batch_size=1, 3 channels, 256x256 resolution
    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=False)
    
    try:
        # Export self-contained ONNX using dynamo=False (avoids external .data file)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['retinal_input'],
            output_names=['stage_logits'],
            dynamo=False
        )
    except Exception as e:
        # Fallback if file is temporarily locked by active inference session
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return output_path, file_size_mb
        raise e
    
    # Verify ONNX model validity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return output_path, file_size_mb

class EdgeONNXInferenceEngine:
    """
    Ultra-lightweight edge inference engine powered by ONNX Runtime CPU.
    """
    def __init__(self, model_path=ONNX_PATH):
        if not os.path.exists(model_path):
            export_dr_model_to_onnx(model_path)
            
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def predict(self, img_rgb):
        """
        Runs preprocessed fundus image through the ONNX runtime.
        Returns:
        - probs: Array of 5 stage probabilities
        - latency_ms: Inference execution time in milliseconds
        """
        # Resize to 256x256 and normalize
        resized = cv2_resize(img_rgb, (256, 256))
        tensor = resized.astype(np.float32) / 255.0
        # Transpose HWC -> CHW and add batch dimension NCHW
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        # Benchmark execution time
        t_start = time.perf_counter()
        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})
        t_end = time.perf_counter()
        
        latency_ms = round((t_end - t_start) * 1000.0, 2)
        logits = raw_outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs, latency_ms

def cv2_resize(img, size):
    import cv2
    return cv2.resize(img, size)

# Pre-export on import if not exists
if not os.path.exists(ONNX_PATH):
    try:
        export_dr_model_to_onnx(ONNX_PATH)
    except Exception as e:
        pass
