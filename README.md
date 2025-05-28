# Real-Time Object Detection for Edge Computing-Based Agricultural Automation: YOLOX vs YOLOv12

This repository contains the implementation code and profiling data for the comparative analysis of YOLOX and YOLOv12 object detection models on NVIDIA Jetson AGX Orin for agricultural automation systems.

## Paper Information

**Title:** Real-Time Object Detection for Edge Computing-Based Agricultural Automation: A Comparative Analysis of YOLOX and YOLOv12 Architectures and Performance

**Authors:** Joonam Kim, Giryon Kim, Rena Yoshitoshi, Kenichi Tokuda

**Affiliation:** National Agricultural and Food Research Organization, Research Center for Agricultural Robotics

**Submitted to:** Sensors (MDPI)

## Repository Contents

```
YOLOXvsYOLOv12/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── Model_test_YOLOX.py           # YOLOX inference and evaluation
├── Model_test_YOLOv12.py         # YOLOv12 inference and evaluation
├── export_trt_YOLOv12.py         # YOLOv12 TensorRT conversion (custom)
├── train_YOLOv12.py              # YOLOv12 training script (custom)
├── t_test_YOLOX.py               # Statistical analysis for YOLOX
├── t_test_YOLOv12.py             # Statistical analysis for YOLOv12
├── video_test_YOLOX.py           # Video inference for YOLOX
├── video_test_YOLOv12.py         # Video inference for YOLOv12
├── join_files.bat                # File reconstruction script (Windows)
├── YOLOX_trt_nsight_part_*.zip   # Split YOLOX profiling data (5 parts)
└── YOLOv12_trt.7z                # YOLOv12 profiling data
```

## Key Results

### Performance Summary (Jetson AGX Orin)
- **YOLOX**: 36 FPS, 14.57 frames/kJ energy efficiency
- **YOLOv12**: 24 FPS, 4.79 frames/kJ energy efficiency
- **YOLOv12**: Superior recall for challenging classes (soil clod: 0.667 vs 0.381)
- **YOLOX**: Better overall precision and real-time performance

## Hardware Configuration

**YOLOX Evaluation:**
- Platform: NVIDIA Jetson AGX Orin 64GB
- JetPack: 5.1.2
- CUDA: 11.4, TensorRT: 8.2

**YOLOv12 Evaluation:**
- Platform: NVIDIA Jetson AGX Orin 64GB  
- JetPack: 6.2
- CUDA: 12.2, TensorRT: 8.6

## Code Usage

### Training (YOLOv12)
```bash
python train_YOLOv12.py --data your_dataset.yaml --epochs 300
```

### TensorRT Conversion (YOLOv12)
```bash
python export_trt_YOLOv12.py --weights yolov12.pt --imgsz 640
```

### Model Evaluation
```bash
# YOLOX evaluation
python Model_test_YOLOX.py --model your_model.pth --data test_data/

# YOLOv12 evaluation  
python Model_test_YOLOv12.py --model your_model.pt --data test_data/
```

### Video Testing
```bash
# Real-time video inference
python video_test_YOLOX.py --source video.mp4
python video_test_YOLOv12.py --source video.mp4
```

### Statistical Analysis
```bash
# Performance statistics with 30-trial validation
python t_test_YOLOX.py --results results.json
python t_test_YOLOv12.py --results results.json
```

## NVIDIA Nsight Profiling Data

### Extracting Profiling Files

**YOLOX Data (Split Files):**
```bash
# Windows
join_files.bat

# Linux/Mac
cat YOLOX_trt_nsight_part_*.zip > YOLOX_combined.zip
unzip YOLOX_combined.zip
```

**YOLOv12 Data:**
```bash
# Extract 7z file
7z x YOLOv12_trt.7z
```

### Data Contents
- Complete NVIDIA Nsight profiling sessions (.nsys-rep format)
- CUDA kernel execution analysis
- Memory transfer patterns and efficiency metrics
- Multi-process overhead analysis (YOLOv12)
- System resource utilization data

### Viewing Data
1. Install NVIDIA Nsight Systems
2. Open extracted .nsys-rep files
3. Analyze execution patterns and performance bottlenecks

## Key Findings

### Computational Efficiency
- YOLOX: 91.1% kernel utilization, 96.1% Host-to-Device transfers
- YOLOv12: 98.9% kernel utilization, 71.5% Host-to-Device transfers
- YOLOv12 multi-process architecture creates significant IPC overhead

### Detection Performance
- YOLOX favors precision (good for minimizing false positives)
- YOLOv12 shows better recall for underrepresented classes
- Data imbalance significantly affects both models differently

## Data and Model Availability

### Available Resources
- Complete implementation code and evaluation scripts
- NVIDIA Nsight profiling data (total ~226MB compressed)
- Statistical analysis frameworks and methodologies
- Custom YOLOv12 training and TensorRT conversion scripts

### Restricted Resources
- Agricultural dataset (institutional data protection policies)
- Trained model weights (institutional policies)

### Reproducibility
The provided code enables full methodological reproducibility using your own datasets. All scripts support the 30-trial statistical validation methodology described in the paper.

## Citation

```bibtex
@article{kim2025realtime,
  title={Real-Time Object Detection for Edge Computing-Based Agricultural Automation: A Comparative Analysis of YOLOX and YOLOv12 Architectures and Performance},
  author={Kim, Joonam and Kim, Giryon and Yoshitoshi, Rena and Tokuda, Kenichi},
  journal={Sensors},
  year={2025},
  note={Submitted}
}
```

## Contact

**Corresponding Author:** Joonam Kim (kim.joonam510@naro.go.jp)  
**Institution:** National Agricultural and Food Research Organization, Research Center for Agricultural Robotics
