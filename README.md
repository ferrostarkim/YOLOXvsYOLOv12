# YOLOXvsYOLOv12
Evaluation code and preprocessing routines for the paper 'Real-Time Object Detection for Edge Computing-Based Agricultural Automation
# Real-Time Object Detection for Edge Computing-Based Agricultural Automation: YOLOX vs. YOLOv12

This repository contains the code and evaluation scripts for the research paper: "Real-Time Object Detection for Edge Computing-Based Agricultural Automation: A Comparative Analysis of YOLOX and YOLOv12 Architectures and Performance"[cite: 1]. This study provides a comprehensive comparison of YOLOX and YOLOv12 object detection models, specifically tailored for potato harvesting applications on the Jetson AGX Orin platform. [cite: 1, 3]

## Overview

The project focuses on evaluating YOLOX and YOLOv12 for real-time crop and impurity differentiation in agricultural automation. [cite: 9] We analyze architectural differences[cite: 2, 4, 5], performance metrics (speed, accuracy, resource utilization)[cite: 3, 6], and class-specific/size-specific detection capabilities, particularly in data-imbalanced agricultural environments. [cite: 3, 72, 76] The goal is to provide empirically-grounded guidelines for model selection in edge computing scenarios for agricultural robotics. [cite: 7]

**System Architecture:** The target application is an integrated AI potato harvester system where an RGB camera captures images of objects on a conveyor. [cite: 16] An AI edge computer processes these images to classify objects as potatoes or impurities. [cite: 17] Based on this, a PLC controls an air cylinder to sort impurities. [cite: 18, 19] (See Figure 1 in the paper [cite: 15]). This research compares a single-process YOLOX implementation with a multi-process YOLOv12 implementation. [cite: 54, 56, 57, 59]

**Key Findings:**
* YOLOX achieves significantly higher throughput (FPS) on the Jetson AGX Orin. [cite: 3]
* YOLOv12 demonstrates superior detection for challenging classes (e.g., soil clods) and small objects, particularly in data-imbalanced contexts. [cite: 3]
* Architectural differences (YOLOv12's R-ELAN backbone and Area Attention vs. YOLOX's decoupled head and SimOTA) significantly impact precision-recall characteristics. [cite: 4, 5]
* Implementation efficiency (CUDA stream orchestration, single vs. multi-process) plays a critical role in practical performance on edge devices. [cite: 6]

## Repository Contents

This repository includes the following scripts used for training, conversion, evaluation, and application:

* **Dataset Handling & Training:**
    * `train_YOLOv12.py`: Script for training or fine-tuning YOLOv12 models using the Ultralytics framework.
* **Model Conversion:**
    * `export_trt_YOLOv12.py`: Script to convert YOLOv12 `.pt` models to TensorRT `.engine` format.
* **Comprehensive Model Evaluation (Image-based):**
    * `Model_test_YOLOv12.py`: Advanced evaluation script for YOLOv12 TensorRT engines, focusing on detailed metrics (precision, recall, F1 per class/size), optimal threshold searching, and various visualizations.
    * `Model_test_YOLOX.py`: Advanced evaluation script for YOLOX TensorRT engines, refined for thesis analysis with specific metrics like potato misclassification rate and impurity recall, and a multi-stage optimization strategy.
    * `YOLOX_Evaluation.py`: Comprehensive evaluation script for YOLOX TensorRT engines, mirroring the structure and metrics of `Model_test_YOLOv12.py` for consistent comparison.
* **Video Processing & Performance Benchmarking:**
    * `video_test_YOLOv12.py`: Script for benchmarking YOLOv12 TensorRT model performance (FPS, latency) on video input, featuring a multi-process architecture for inference.
    * `video_test_YOLOX.py`: (Modified version based on previous discussion) Script for benchmarking YOLOX (PyTorch or TRT) model performance on video input, simplified for FPS and latency measurement with on-screen visualization.
* **Statistical Analysis:**
    * `t_test_YOLOv12.py`: Script for running multiple YOLOv12 evaluation trials and performing t-tests for statistical significance of metrics.
    * `t_test_YOLOX.py`: Script for running multiple YOLOX evaluation trials and performing statistical analysis (potentially t-tests if `scipy` is available).
* **Application & Benchmarking Utilities:**
    * `benchmark_model.py`: A wrapper script to run and manage YOLOX (PyTorch vs. TensorRT) benchmark evaluations by calling external evaluator and visualizer scripts.
    * `main_YOLOX_v30_paper.py` / `main_YOLOX_v30_triming.py`: Full application scripts for a YOLOX-based potato harvesting system, including Basler camera integration, real-time inference, PLC control logic, and detailed logging.

## Dataset

The dataset comprises 10,000 images (1280x960 pixels) of potatoes and impurities (soil clods, stones) collected from commercial farms in Hokkaido, Japan. [cite: 68, 69] It was split into 80% training, 10% validation, and 10% testing. [cite: 70] The dataset exhibits significant class (Potato: ~60%, Soil Clod: ~20%, Stone: ~20%) and object size imbalance, reflecting real-world agricultural conditions. [cite: 72, 76] (See Figure 3 in the paper [cite: 67]).

## Setup

Detailed setup instructions for YOLOX and YOLOv12 can be found in their respective official repositories. Key dependencies include:
* Python 3.8
* PyTorch
* OpenCV
* Ultralytics (for YOLOv12 scripts)
* YOLOX framework
* TensorRT (and `torch2trt` for YOLOX TRT)
* `loguru`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy` (for some scripts)

Ensure all paths to models, datasets, and experiment files are correctly configured within each script or passed as command-line arguments.

## Usage

Each script can be run independently. Refer to the command-line arguments within each file for specific usage instructions.
Example (YOLOv12 evaluation):
`python Model_test_YOLOv12.py --model_path <path_to_yolov12.engine> --dataset_yaml <path_to_data.yaml>`

Example (YOLOX video test):
`python video_test_YOLOX.py -f <exp_file.py> -c <yolox_model.pth_or_trt> --path <video_file_or_cam_id>`

## Note on Dataset and Full System Code Availability

Due to institutional data protection policies, the full dataset and the complete operational code for the Jetson AGX Orin-based harvester system (as depicted in Figure 1 of the paper) are not publicly available. [cite: 339] However, this repository provides the core scripts for processing video input, performing inference, generating output results using trained models (which users would need to train on their own or similar datasets), and scripts for calculating performance metrics and conducting evaluations as described in the paper. [cite: 340]

## Citation

If you use this work or code, please cite our paper:

```bibtex
@article{Kim2025RealTimeOD,
  title={Real-Time Object Detection for Edge Computing-Based Agricultural Automation: A Comparative Analysis of YOLOX and YOLOv12 Architectures and Performance},
  author={Joonam Kim and Giryon Kim and Rena Yoshitoshi and Kenichi Tokuda},
  journal={Sensors (MDPI)}, # Or the actual journal once accepted/published
  year={2025}, # Or actual year of publication
  
}
