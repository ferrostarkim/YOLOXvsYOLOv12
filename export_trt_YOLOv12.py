# -*- coding: utf-8 -*-
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse 

# --- Basic settings ---
DEFAULT_MODEL_PATH = 'yolov12s.pt'  # Basic model path (Standard)
NANO_MODEL_PATH = 'yolov12n.pt'     # Nano model path
EXPORT_IMG_SIZE = 640             # TensorRT engine input image size (single value or (height, width) tuple)
                                  # (e.g., 640 or (640, 640))
USE_FP16 = True                   # Use FP16 precision
# USE_INT8 = False                # Use INT8 precision (uncomment and set half=False if needed)
WORKSPACE_GB = 4                  # Maximum GPU memory to use for TensorRT build (GB)
DEVICE_ID = 0                     # GPU device ID to use
# --- /Basic settings ---

def export_to_tensorrt(pt_model_path, use_fp16=True, workspace_gb=4, device_id=0, img_size=640):
    """ Convert specified .pt model to TensorRT .engine file """
    model_path_obj = Path(pt_model_path)
    print(f"Loading model from {model_path_obj}...")
    try:
        # Load model
        model = YOLO(model_path_obj)
        print(f"Model '{model_path_obj.name}' loaded successfully.")

        # ONNX conversion is automatically handled by the export function, so remove explicit ONNX conversion
        # (add onnx-related arguments in the export call if needed)

        print("Exporting to TensorRT engine...")
        # Try direct conversion to TensorRT
        model.export(
            format='engine',       # Target format: TensorRT engine
            imgsz=img_size,        # Input image size
            half=use_fp16,         # Use FP16
            # int8=USE_INT8,       # Use INT8 (uncomment and set half=False if needed)
            workspace=workspace_gb,# Memory to use for build
            device=str(device_id), # Device ID (recommended to pass as string)
            batch=1,               # Batch size (usually 1 for real-time inference)
            # dynamic=False,       # Use dynamic input size (False recommended)
            simplify=True,         # Simplify ONNX model (done internally before engine conversion)
            verbose=True           # Print detailed information
        )

        # Check .engine file name (YOLO export function automatically creates a model-based name)
        expected_engine_path = model_path_obj.with_suffix('.engine')
        if expected_engine_path.exists():
             print(f"\nExport successful! TensorRT engine saved to: {expected_engine_path.resolve()}")
        else:
             # The export function may save with a different name, so check the message
             print(f"\nExport process completed for {model_path_obj.name}. Please check the directory for the '.engine' file.")
             print(f"(Expected path: {expected_engine_path})")


    except Exception as e:
        print(f"\nError during export for {model_path_obj.name}: {e}")
        import traceback
        traceback.print_exc()
        print("\nExport failed. Please check the following:")
        print("- Correct ultralytics, PyTorch, CUDA, cuDNN, and TensorRT versions are installed and compatible on your Jetson.")
        print(f"- The specified model path '{pt_model_path}' exists.")
        print("- Sufficient GPU memory is available (try reducing WORKSPACE_GB or closing other GPU-intensive applications).")
        print("- Ensure the model architecture is compatible with TensorRT export.")

if __name__ == '__main__':
    # Set command line argument parser
    parser = argparse.ArgumentParser(description='Export YOLOv12 .pt model to TensorRT engine.')
    parser.add_argument('--nano', action='store_true', help='Export the nano version (yolov12n.pt) instead of the standard (yolov12s.pt).')
    args = parser.parse_args()

    # Determine the model path to convert based on the --nano flag
    if args.nano:
        model_to_export = NANO_MODEL_PATH
        print("Nano model selected for export.")
    else:
        model_to_export = DEFAULT_MODEL_PATH
        print("Standard model selected for export.")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(DEVICE_ID)}")
        # Pass the selected model path and settings to the function for execution
        export_to_tensorrt(
            pt_model_path=model_to_export,
            use_fp16=USE_FP16,
            workspace_gb=WORKSPACE_GB,
            device_id=DEVICE_ID,
            img_size=EXPORT_IMG_SIZE
        )
    else:
        print("Error: CUDA is not available. TensorRT export requires a CUDA-enabled GPU.")
