#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOv12 t-test script (using prepared dataset, FPS measurement excluded)

import argparse
import os
import time
import numpy as np
import torch
import yaml
import json
import sys
from pathlib import Path
from collections import defaultdict
from loguru import logger
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
from scipy import stats as scipy_stats  

def make_parser():
    parser = argparse.ArgumentParser("YOLOv12 t-test Evaluation")
    parser.add_argument("-c", "--ckpt", 
                       default="yolov12s.pt", 
                       type=str, 
                       help="YOLOv12 model path")
    parser.add_argument("-d", "--data", 
                       default="dataset/data.yaml", 
                       type=str, 
                       help="Dataset YAML file path (prepared data.yaml)")
    parser.add_argument("--device", 
                       default="0", 
                       type=str, 
                       help="Device to run on (e.g., 0 for CUDA device 0)")
    parser.add_argument("--conf", 
                       default=0.25, 
                       type=float, 
                       help="Confidence threshold")
    parser.add_argument("--iou", 
                       default=0.45, 
                       type=float, 
                       help="IoU threshold for NMS")
    parser.add_argument("--img-size", 
                       default=640, 
                       type=int, 
                       help="Test image size")
    parser.add_argument("--batch-size", 
                       default=16, 
                       type=int, 
                       help="Batch size for evaluation")
    parser.add_argument("--num-runs", 
                       type=int, 
                       default=30,
                       help="Number of evaluation runs for t-test analysis")
    parser.add_argument("--output-dir", 
                       type=str, 
                       default="yolov12_ttest_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--save-plots", 
                       action="store_true",
                       help="Generate and save plots for metrics")
    return parser

def run_single_evaluation(model, args, run_id):
    """Single evaluation run"""
    logger.info(f"Running evaluation {run_id+1}/{args.num_runs}")
    
    # Evaluation
    results = model.val(
        data=args.data,
        split='test',  # use test data
        batch=args.batch_size,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,  # simplified output
        plots=run_id == 0,  # create plots only on first run
        save_json=False,
        save_hybrid=False
    )
    
    # Extract result data
    metrics = {}
    
    # Extract metrics from results
    if hasattr(results, 'results_dict'):
        for k, v in results.results_dict.items():
            if isinstance(v, (int, float, np.number)):
                metrics[k] = float(v)
    
    # If class-specific metrics exist
    class_metrics = {}
    if hasattr(results, 'names') and hasattr(results, 'metrics'):
        for i, name in enumerate(results.names):
            if hasattr(results.metrics, 'class_result') and i < len(results.metrics.class_result):
                class_data = results.metrics.class_result[i]
                class_metrics[name] = {
                    "precision": float(class_data[0]),
                    "recall": float(class_data[1]),
                    "map50": float(class_data[2]),
                    "map50-95": float(class_data[3])
                }
    
    # Evaluation result
    eval_result = {
        "run_id": run_id + 1,
        "metrics": metrics,
        "class_metrics": class_metrics
    }
    
    return eval_result

def run_statistical_evaluation(args):
    """여러 번의 평가를 실행하고 통계 분석"""
    # Create output directory        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output_dir, f"ttest_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading YOLOv12 model from {args.ckpt}")
    model = YOLO(args.ckpt)
    
    # All results save list
    all_results = []
    
    # Repeat evaluation
    for run_id in range(args.num_runs):
        eval_result = run_single_evaluation(model, args, run_id)
        all_results.append(eval_result)
        
        # Save each run result
        with open(os.path.join(result_dir, f"run_{run_id+1}.json"), "w") as f:
            json.dump(eval_result, f, indent=2)
    
    # Save all results
    with open(os.path.join(result_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Collect metrics statistics
    metric_values = defaultdict(list)
    for result in all_results:
        for metric_name, metric_value in result["metrics"].items():
            metric_values[metric_name].append(metric_value)
    
    # Collect class-specific metrics
    class_metric_values = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        for class_name, class_data in result.get("class_metrics", {}).items():
            for metric_name, metric_value in class_data.items():
                class_metric_values[class_name][metric_name].append(metric_value)
    
    # Statistical analysis results
    stats_results = {
        "metrics": {},
        "class_metrics": {}
    }
    
    # Calculate total metrics statistics
    for metric_name, values in metric_values.items():
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        
        # Single sample t-test (null hypothesis: mean = 0)
        t_stat, p_val = scipy_stats.ttest_1samp(values_array, 0)
        
        # Calculate 95% confidence interval
        ci_lower, ci_upper = scipy_stats.t.interval(
            0.95,
            len(values_array)-1,
            loc=mean_val,
            scale=scipy_stats.sem(values_array)
        )
        
        stats_results["metrics"][metric_name] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(min_val),
            "max": float(max_val),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "values": values
        }
    
    # Calculate class-specific metrics statistics
    for class_name, metrics in class_metric_values.items():
        stats_results["class_metrics"][class_name] = {}
        
        for metric_name, values in metrics.items():
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            
            # Single sample t-test (null hypothesis: mean = 0)
            t_stat, p_val = scipy_stats.ttest_1samp(values_array, 0)
            
            # Calculate 95% confidence interval
            ci_lower, ci_upper = scipy_stats.t.interval(
                0.95,
                len(values_array)-1,
                loc=mean_val,
                scale=scipy_stats.sem(values_array)
            )
            
            stats_results["class_metrics"][class_name][metric_name] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(min_val),
                "max": float(max_val),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "values": values
            }
    
    # Save statistical results
    with open(os.path.join(result_dir, "statistical_analysis.json"), "w") as f:
        json.dump(stats_results, f, indent=2)
    
    # Create text report
    with open(os.path.join(result_dir, "statistical_report.txt"), "w") as f:
        f.write(f"YOLOv12 Statistical Evaluation Report (t-test)\n")
        f.write(f"===========================================\n\n")
        f.write(f"Model: {args.ckpt}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Number of runs: {args.num_runs}\n\n")
        
        # Report overall metrics
        f.write("Overall Metrics\n")
        f.write("--------------\n\n")
        
        for metric_name, stats in sorted(stats_results["metrics"].items()):
            f.write(f"{metric_name}:\n")
            f.write(f"  Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
            f.write(f"  95% CI: [{stats['ci_95_lower']:.6f}, {stats['ci_95_upper']:.6f}]\n")
            f.write(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}\n")
            f.write(f"  t-statistic: {stats['t_statistic']:.4f}, p-value: {stats['p_value']:.6f}\n\n")
        
        # Report class-specific metrics
        f.write("\nClass-Specific Metrics\n")
        f.write("---------------------\n\n")
        
        for class_name, metrics in sorted(stats_results["class_metrics"].items()):
            f.write(f"Class: {class_name}\n")
            f.write(f"{'-' * (7 + len(class_name))}\n\n")
            
            for metric_name, stats in sorted(metrics.items()):
                f.write(f"  {metric_name}:\n")
                f.write(f"    Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
                f.write(f"    95% CI: [{stats['ci_95_lower']:.6f}, {stats['ci_95_upper']:.6f}]\n")
                f.write(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}\n")
                f.write(f"    t-statistic: {stats['t_statistic']:.4f}, p-value: {stats['p_value']:.6f}\n\n")
    
    # Create plots (if requested)
    if args.save_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plots_dir = os.path.join(result_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Metric-wise histogram and distribution plots
            for metric_name, stats in stats_results["metrics"].items():
                plt.figure(figsize=(10, 6))
                sns.histplot(stats["values"], kde=True)
                plt.axvline(stats["mean"], color='r', linestyle='--', label=f'Mean: {stats["mean"]:.4f}')
                plt.axvline(stats["ci_95_lower"], color='g', linestyle=':', label=f'95% CI Lower: {stats["ci_95_lower"]:.4f}')
                plt.axvline(stats["ci_95_upper"], color='g', linestyle=':', label=f'95% CI Upper: {stats["ci_95_upper"]:.4f}')
                plt.title(f'Distribution of {metric_name} over {args.num_runs} runs')
                plt.xlabel(metric_name)
                plt.ylabel('Frequency')
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f"{metric_name}_distribution.png"), dpi=300)
                plt.close()
            
            # Class-wise comparison bar plots for key metrics
            for metric_name in ["precision", "recall", "map50"]:
                class_names = []
                means = []
                stds = []
                
                for class_name, metrics in stats_results["class_metrics"].items():
                    if metric_name in metrics:
                        class_names.append(class_name)
                        means.append(metrics[metric_name]["mean"])
                        stds.append(metrics[metric_name]["std"])
                
                if class_names:
                    plt.figure(figsize=(12, 6))
                    plt.bar(class_names, means, yerr=stds, capsize=5)
                    plt.title(f'Class-wise {metric_name} comparison')
                    plt.xlabel('Class')
                    plt.ylabel(metric_name)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"class_{metric_name}_comparison.png"), dpi=300)
                    plt.close()
            
            logger.info(f"Plots saved to {plots_dir}")
        except ImportError:
            logger.warning("Could not generate plots. Make sure matplotlib and seaborn are installed.")
    
    # Return important metric average value
    key_metric = "metrics/mAP50(B)"
    if key_metric in stats_results["metrics"]:
        return stats_results["metrics"][key_metric]["mean"]
    else:
        # Find alternative metric
        for metric_name in stats_results["metrics"]:
            if "map50" in metric_name.lower():
                return stats_results["metrics"][metric_name]["mean"]
        return 0.0

def main():
    args = make_parser().parse_args()
    
    # Logger settings
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(os.path.join(args.output_dir, "yolov12_ttest.log"), rotation="10 MB", level="INFO")
    
    logger.info("YOLOv12 t-test Evaluation")
    logger.info(f"Arguments: {args}")
    
    try:
        # Check dataset YAML file
        if not os.path.exists(args.data):
            logger.error(f"Dataset YAML file not found: {args.data}")
            return 1
        
        # Run statistical evaluation
        map50 = run_statistical_evaluation(args)
        logger.info(f"Evaluation completed successfully! Average mAP@50: {map50:.4f}")
        return 0
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())