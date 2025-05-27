# -*- coding: utf-8 -*-
# YOLOv12 Advanced Evaluation Script for Agriculture (TensorRT ENGINE ONLY - v5.1-trt)
# Usage: python3 yolov12_evaluation_v5_trt.py
# Changes: v5-trt + Fixed CM plot error, Enhanced comparison visualization with TP/FP/FN labels.

import os
import sys
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time
import pickle
import multiprocessing
import functools
import json
import seaborn as sns
import matplotlib

# Ensure matplotlib backend is suitable for non-interactive environments if needed
# matplotlib.use('Agg') # Use 'Agg' backend to avoid GUI errors on servers without display

# --- Configuration (Hardcoded Parameters) ---

# Model & Data Paths
MODEL_PATH = 'yolov12s.engine'  # <<<--- 수정: 실제 TensorRT 엔진 파일 (.engine) 경로로 변경하세요.
DATASET_YAML = 'dataset/data.yaml' # <<<--- 수정: 실제 데이터셋 YAML 파일 경로로 변경하세요.

# Output Paths
OUTPUT_DIR = 'yolov12_trt_eval_results_v5_1_trt' # Changed output dir name
RAW_DATA_FILENAME = "raw_eval_data.pkl"
RESULTS_FILENAME = "all_evaluation_results_v5_1_trt.csv" # Changed results filename
BEST_RESULT_FILENAME = "best_evaluation_result_v5_1_trt.json" # Filename for best result
COMPARISON_VIS_DIR = "comparison_visuals_enhanced" # Subdir for comparison images
ANALYSIS_PLOTS_DIR = "analysis_plots" # Subdir for analysis plots

# Class IDs (Ensure these match your dataset YAML)
POTATO_CLS_IDS = {0}  # <<<--- 수정: 실제 감자 클래스 ID로 변경 (Set 형태), 예: {0}
IMPURITY_CLS_IDS = {1, 2} # <<<--- 수정: 실제 불순물 클래스 ID들로 변경 (Set 형태), 예: {1, 2}
# 가정: 0=baresho, 1=dokai, 2=ishi

# Evaluation Parameters
IOU_THRESHOLD = 0.5 # IoU threshold for TP/FP matching

# Size Bins Definition
SIZE_BINS = [
    (0, 3000), (3000, 6000), (6000, 9000), (9000, 12000), (12000, 15000),
    (15000, 18000), (18000, 21000), (21000, 24000), (24000, float('inf'))
]
def get_size_bin_label(area, bins):
    for idx, (lower, upper) in enumerate(bins):
        if lower <= area < upper:
            if upper == float('inf'): return f">={int(lower/1000)}k"
            return f"{int(lower/1000)}k-{int(upper/1000)}k"
    return "Unknown"

DEVICE = "cuda:0"

# Standard Thresholds for Comparison Visualization
STANDARD_POTATO_CONF = 0.5
STANDARD_IMPURITY_CONF = 0.5
STANDARD_POTATO_NMS = 0.45
STANDARD_IMPURITY_NMS = 0.45

# Threshold Ranges for Analysis
POTATO_CONF_RANGE = np.round(np.arange(0.45, 0.86, 0.1), 3)
IMPURITY_CONF_RANGE = np.round(np.arange(0.2, 0.66, 0.1), 3)
POTATO_NMS_RANGE = np.round(np.arange(0.35, 0.56, 0.1), 3)
IMPURITY_NMS_RANGE = np.round(np.arange(0.35, 0.56, 0.1), 3)

# System Goal Targets (Applied to OVERALL metrics)
TARGET_POTATO_PRECISION = 0.990
TARGET_IMPURITY_RECALL = 0.600

# --- Ultralytics YOLO import ---
try: from ultralytics import YOLO
except ImportError: logger.error("Ultralytics not found. pip install ultralytics"); sys.exit(1)

# --- Utility Functions ---
# (calculate_iou, get_box_area, load_yolo_annotation - Keep from previous)
def calculate_iou(box1, box2):
    if not (len(box1) >= 4 and len(box2) >= 4): return 0.0
    if box1[2] <= box1[0] or box1[3] <= box1[1] or box2[2] <= box2[0] or box2[3] <= box2[1]: return 0.0
    x1_inter=max(box1[0],box2[0]); y1_inter=max(box1[1],box2[1]); x2_inter=min(box1[2],box2[2]); y2_inter=min(box1[3],box2[3])
    width_inter=max(0,x2_inter-x1_inter); height_inter=max(0,y2_inter-y1_inter); area_inter=width_inter*height_inter
    area_box1=(box1[2]-box1[0])*(box1[3]-box1[1]); area_box2=(box2[2]-box2[0])*(box2[3]-box2[1])
    if area_box1<=0 or area_box2<=0: return 0.0
    area_union=area_box1+area_box2-area_inter; iou=area_inter/area_union if area_union > 0 else 0; return float(iou)
def get_box_area(box):
    if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]: return 0.0
    return float((box[2] - box[0]) * (box[3] - box[1]))
def load_yolo_annotation(annotation_path, img_width, img_height, class_names):
    boxes = []
    if not os.path.exists(annotation_path): return np.array([])
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split();
                if len(parts)!=5: continue
                class_id, x_c, y_c, w_n, h_n = map(float, parts); class_id = int(class_id)
                if class_id < 0 or class_id >= len(class_names): continue
                x1=(x_c-w_n/2)*img_width; y1=(y_c-h_n/2)*img_height; x2=(x_c+w_n/2)*img_width; y2=(y_c+h_n/2)*img_height
                x1=max(0.0,x1); y1=max(0.0,y1); x2=min(img_width,x2); y2=min(img_height,y2)
                if x1>=x2 or y1>=y2: continue
                boxes.append([x1, y1, x2, y2, class_id])
    except Exception as e: logger.error(f"Read/parse {annotation_path}: {e}"); return np.array([])
    return np.array(boxes, dtype=np.float32)


# --- Raw Data Handling ---
# (Keep save_raw_data, load_raw_data from previous)
def save_raw_data(data, filepath):
    logger.info(f"Saving raw data to {filepath}...");
    try:
        with open(filepath, 'wb') as f: pickle.dump(data, f); logger.info("Raw data saved.")
    except Exception as e: logger.error(f"Save raw data failed: {e}")
def load_raw_data(filepath):
    logger.info(f"Loading raw data from {filepath}...");
    if not os.path.exists(filepath): logger.warning(f"Raw data not found: {filepath}"); return None
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f); logger.info("Raw data loaded."); return data
    except Exception as e: logger.error(f"Load raw data failed: {e}"); return None

# --- NMS Function ---
# (Keep apply_nms_to_detections from previous)
def apply_nms_to_detections(detections, nms_threshold):
    if not detections: return []
    try:
        boxes = np.array([det[:4] for det in detections], dtype=np.float32)
        scores = np.array([det[5] for det in detections], dtype=np.float32)
    except (IndexError, ValueError, TypeError): return []
    keep_indices = []; order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]; keep_indices.append(i);
        if order.size == 1: break
        if i >= len(boxes): break
        xx1=np.maximum(boxes[i,0],boxes[order[1:],0]); yy1=np.maximum(boxes[i,1],boxes[order[1:],1])
        xx2=np.minimum(boxes[i,2],boxes[order[1:],2]); yy2=np.minimum(boxes[i,3],boxes[order[1:],3])
        w=np.maximum(0.0,xx2-xx1); h=np.maximum(0.0,yy2-yy1); inter=w*h
        area_i=(boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]); area_others=(boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        union=area_i+area_others-inter+1e-6; ovr=inter/union; ovr[union<=1e-6]=0
        inds=np.where(ovr<=nms_threshold)[0]; order=order[inds+1]
    return [detections[idx] for idx in keep_indices if idx < len(detections)]


# --- Combined Confidence and NMS Filtering Function ---
# (Keep apply_confidence_and_nms from previous)
def apply_confidence_and_nms(raw_detections,
                            potato_conf_thresh, impurity_conf_thresh,
                            potato_nms_thresh, impurity_nms_thresh,
                            potato_cls_ids, impurity_cls_ids):
    conf_filtered_potato = []; conf_filtered_impurity = []
    for det in raw_detections:
        if len(det) < 6: continue
        cls_id = int(det[4]); score = float(det[5])
        if cls_id in potato_cls_ids:
            if score >= potato_conf_thresh: conf_filtered_potato.append(det)
        elif cls_id in impurity_cls_ids:
            if score >= impurity_conf_thresh: conf_filtered_impurity.append(det)
    nms_filtered_potato = apply_nms_to_detections(conf_filtered_potato, potato_nms_thresh)
    nms_filtered_impurity = apply_nms_to_detections(conf_filtered_impurity, impurity_nms_thresh)
    final_detections = nms_filtered_potato + nms_filtered_impurity
    return final_detections

# --- Comprehensive Evaluation Logic ---
# (Keep evaluate_precision_recall_comprehensive from previous)
def evaluate_precision_recall_comprehensive(detections, ground_truths, num_classes, iou_threshold,
                                            potato_cls_ids, impurity_cls_ids, size_bins, class_names_map):
    stats = {
        'overall_potato': {'tp': 0,'fp': 0,'fn': 0}, 'overall_impurity': {'tp': 0,'fp': 0,'fn': 0},
        'per_class': defaultdict(lambda: {'tp': 0,'fp': 0,'fn': 0}),
        'per_size_bin': defaultdict(lambda: {'tp': 0,'fp': 0,'fn': 0}),
        'per_class_size_bin': defaultdict(lambda: defaultdict(lambda: {'tp': 0,'fp': 0,'fn': 0}))
    }
    size_bin_labels = {i: get_size_bin_label(0, size_bins) for i in range(len(size_bins))}
    gt_boxes_list = ground_truths.tolist() if isinstance(ground_truths, np.ndarray) else ground_truths; det_boxes_list = detections
    if not gt_boxes_list and not det_boxes_list: return stats
    gt_info = []
    for i, gt in enumerate(gt_boxes_list):
        if len(gt) < 5: continue
        gt_box=gt[:4]; gt_cls=int(gt[4]); gt_area=get_box_area(gt_box); is_potato=gt_cls in potato_cls_ids; is_impurity=gt_cls in impurity_cls_ids; size_bin_idx=-1
        for idx,(lower,upper) in enumerate(size_bins):
             if lower<=gt_area<upper: size_bin_idx=idx; break
        gt_info.append({'index': i,'box': gt_box,'class': gt_cls,'area': gt_area,'is_potato': is_potato,'is_impurity': is_impurity,'size_bin_idx': size_bin_idx,'matched': False})
    det_boxes_list.sort(key=lambda x: x[5], reverse=True); det_matched_status=[{'matched': False,'matched_gt_idx':-1} for _ in range(len(det_boxes_list))]
    for det_idx, det in enumerate(det_boxes_list):
        det_box=det[:4]; det_cls=int(det[4]); det_area=get_box_area(det_box); is_potato_pred=det_cls in potato_cls_ids; is_impurity_pred=det_cls in impurity_cls_ids; det_size_bin_idx=-1
        for idx,(lower,upper) in enumerate(size_bins):
             if lower<=det_area<upper: det_size_bin_idx=idx; break
        best_iou=0.0; best_gt_info_idx=-1
        for gt_info_idx, gt in enumerate(gt_info):
             if gt['class']==det_cls and not gt['matched']:
                 iou=calculate_iou(det_box,gt['box']);
                 if iou>=iou_threshold and iou>best_iou: best_iou=iou; best_gt_info_idx=gt_info_idx
        if best_gt_info_idx!=-1:
            matched_gt=gt_info[best_gt_info_idx]; matched_gt['matched']=True; det_matched_status[det_idx]['matched']=True; det_matched_status[det_idx]['matched_gt_idx']=matched_gt['index']
            gt_class=matched_gt['class']; gt_size_bin_idx=matched_gt['size_bin_idx']
            stats['per_class'][gt_class]['tp']+=1
            if gt_size_bin_idx!=-1: stats['per_size_bin'][gt_size_bin_idx]['tp']+=1; stats['per_class_size_bin'][gt_class][gt_size_bin_idx]['tp']+=1
            if matched_gt['is_potato']: stats['overall_potato']['tp']+=1
            if matched_gt['is_impurity']: stats['overall_impurity']['tp']+=1
    for det_idx, det in enumerate(det_boxes_list):
        if not det_matched_status[det_idx]['matched']:
            det_cls=int(det[4]); det_area=get_box_area(det[:4]); det_size_bin_idx=-1
            for idx,(lower,upper) in enumerate(size_bins):
                 if lower<=det_area<upper: det_size_bin_idx=idx; break
            stats['per_class'][det_cls]['fp']+=1
            if det_size_bin_idx!=-1: stats['per_size_bin'][det_size_bin_idx]['fp']+=1; stats['per_class_size_bin'][det_cls][det_size_bin_idx]['fp']+=1
            is_potato_pred=det_cls in potato_cls_ids; is_impurity_pred=det_cls in impurity_cls_ids
            if is_potato_pred: stats['overall_potato']['fp']+=1
            if is_impurity_pred: stats['overall_impurity']['fp']+=1
    for gt in gt_info:
        if not gt['matched']:
            gt_class=gt['class']; gt_size_bin_idx=gt['size_bin_idx']
            stats['per_class'][gt_class]['fn']+=1
            if gt_size_bin_idx!=-1: stats['per_size_bin'][gt_size_bin_idx]['fn']+=1; stats['per_class_size_bin'][gt_class][gt_size_bin_idx]['fn']+=1
            if gt['is_potato']: stats['overall_potato']['fn']+=1
            if gt['is_impurity']: stats['overall_impurity']['fn']+=1
    return stats


# --- Calculate Derived Metrics ---
# (Keep calculate_derived_metrics from previous)
def calculate_derived_metrics(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'tp': tp, 'fp': fp, 'fn': fn}


# --- Worker Function for Parallel Processing ---
# (Keep evaluate_combination from previous)
# --- Worker Function for Parallel Processing (Modified) ---
def evaluate_combination(combo, all_raw_detections, all_ground_truths,
                         potato_cls_ids, impurity_cls_ids, num_classes, iou_threshold,
                         size_bins, class_names_map):
    """
    Evaluates a single combination of thresholds, ensuring all size bin columns are initialized.
    """
    potato_conf, impurity_conf, potato_nms, impurity_nms = combo
    # Round thresholds for consistency
    potato_conf = round(potato_conf, 3)
    impurity_conf = round(impurity_conf, 3)
    potato_nms = round(potato_nms, 3)
    impurity_nms = round(impurity_nms, 3)

    # --- 1. Aggregate TP/FP/FN counts across all images for this combination ---
    total_stats = {
        'overall_potato': {'tp': 0, 'fp': 0, 'fn': 0},
        'overall_impurity': {'tp': 0, 'fp': 0, 'fn': 0},
        'per_class': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
        'per_size_bin': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
        'per_class_size_bin': defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
    }

    for img_id, raw_data in all_raw_detections.items():
        raw_dets = raw_data.get('detections', [])
        gt_boxes = all_ground_truths.get(img_id, np.array([]))

        # Apply confidence and NMS filtering for this specific combination
        filtered_dets = apply_confidence_and_nms(
            raw_dets, potato_conf, impurity_conf, potato_nms, impurity_nms,
            potato_cls_ids, impurity_cls_ids
        )

        # Evaluate metrics for the current image
        img_stats = evaluate_precision_recall_comprehensive(
            filtered_dets, gt_boxes, num_classes, iou_threshold,
            potato_cls_ids, impurity_cls_ids, size_bins, class_names_map
        )

        # Aggregate counts into total_stats
        for category, metrics in img_stats.items():
            if category in ['overall_potato', 'overall_impurity']:
                total_stats[category]['tp'] += metrics['tp']
                total_stats[category]['fp'] += metrics['fp']
                total_stats[category]['fn'] += metrics['fn']
            elif category == 'per_class':
                for cls_id, class_metrics in metrics.items():
                    total_stats[category][cls_id]['tp'] += class_metrics['tp']
                    total_stats[category][cls_id]['fp'] += class_metrics['fp']
                    total_stats[category][cls_id]['fn'] += class_metrics['fn']
            elif category == 'per_size_bin':
                for bin_idx, bin_metrics in metrics.items():
                    total_stats[category][bin_idx]['tp'] += bin_metrics['tp']
                    total_stats[category][bin_idx]['fp'] += bin_metrics['fp']
                    total_stats[category][bin_idx]['fn'] += bin_metrics['fn']
            elif category == 'per_class_size_bin':
                for cls_id, class_bins in metrics.items():
                    for bin_idx, bin_metrics in class_bins.items():
                        total_stats[category][cls_id][bin_idx]['tp'] += bin_metrics['tp']
                        total_stats[category][cls_id][bin_idx]['fp'] += bin_metrics['fp']
                        total_stats[category][cls_id][bin_idx]['fn'] += bin_metrics['fn']

    # --- 2. Initialize results dictionary and add basic/overall metrics ---
    results = {
        'potato_conf': potato_conf,
        'impurity_conf': impurity_conf,
        'potato_nms': potato_nms,
        'impurity_nms': impurity_nms
    }

    # Calculate and add overall metrics
    overall_potato = calculate_derived_metrics(**total_stats['overall_potato'])
    overall_impurity = calculate_derived_metrics(**total_stats['overall_impurity'])
    results.update({f'overall_potato_{k}': v for k, v in overall_potato.items()})
    results.update({f'overall_impurity_{k}': v for k, v in overall_impurity.items()})

    # Calculate and add per-class overall metrics
    for cls_id, counts in total_stats['per_class'].items():
        if cls_id in class_names_map:
            class_name = class_names_map[cls_id]
            safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
            metrics = calculate_derived_metrics(**counts)
            results.update({f'class_{safe_class_name}_{k}': v for k, v in metrics.items()})

    # --- 3. ***MODIFICATION START***: Initialize ALL size bin columns with placeholders ---
    metrics_placeholder = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

    # Initialize overall size bin columns
    for bin_idx, size_bin in enumerate(size_bins):
        size_label = get_size_bin_label(size_bin[0], size_bins)
        safe_size_label = size_label.replace('<','lt').replace('>=','gte').replace('-','_').replace(' ','')
        results.update({f'size_{safe_size_label}_{k}': v for k, v in metrics_placeholder.items()})

    # Initialize class-specific size bin columns
    for cls_id, cls_name in class_names_map.items():
        safe_cls_name = "".join(c if c.isalnum() else "_" for c in cls_name)
        for bin_idx, size_bin in enumerate(size_bins):
            size_label = get_size_bin_label(size_bin[0], size_bins)
            safe_size_label = size_label.replace('<','lt').replace('>=','gte').replace('-','_').replace(' ','')
            results.update({f'class_{safe_cls_name}_size_{safe_size_label}_{k}': v for k, v in metrics_placeholder.items()})
    # --- ***MODIFICATION END*** ---

    # --- 4. Calculate and update metrics for bins with ACTUAL data (overwriting placeholders) ---
    # Calculate and add per_size_bin metrics (overall classes)
    for bin_idx, counts in total_stats['per_size_bin'].items():
        # Check if bin_idx is valid before accessing size_bins
        if 0 <= bin_idx < len(size_bins):
            size_label = get_size_bin_label(size_bins[bin_idx][0], size_bins)
            safe_size_label = size_label.replace('<','lt').replace('>=','gte').replace('-','_').replace(' ','')
            metrics = calculate_derived_metrics(**counts)
            # Update results, overwriting the placeholders if counts exist
            results.update({f'size_{safe_size_label}_{k}': v for k, v in metrics.items()})
        else:
            logger.warning(f"Invalid bin_idx {bin_idx} encountered in per_size_bin stats for combo {combo}. Skipping.")

    # Calculate and add per_class_size_bin metrics
    for cls_id, class_bins in total_stats['per_class_size_bin'].items():
        if cls_id in class_names_map:
            class_name = class_names_map[cls_id]
            safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
            for bin_idx, counts in class_bins.items():
                 # Check if bin_idx is valid before accessing size_bins
                if 0 <= bin_idx < len(size_bins):
                    size_label = get_size_bin_label(size_bins[bin_idx][0], size_bins)
                    safe_size_label = size_label.replace('<','lt').replace('>=','gte').replace('-','_').replace(' ','')
                    metrics = calculate_derived_metrics(**counts)
                    # Update results, overwriting the placeholders if counts exist
                    results.update({f'class_{safe_class_name}_size_{safe_size_label}_{k}': v for k, v in metrics.items()})
                else:
                    logger.warning(f"Invalid bin_idx {bin_idx} encountered in per_class_size_bin stats for class {cls_id}, combo {combo}. Skipping.")

    # --- 5. Calculate specific small object F1 metric ---
    # Uses the potentially updated stats from total_stats
    small_bin_overall_metrics = calculate_derived_metrics(**total_stats['per_size_bin'].get(0, {'tp': 0, 'fp': 0, 'fn': 0}))
    results['small_overall_f1'] = small_bin_overall_metrics['f1']

    return results

# --- Visualization ---

# FIXED: visualize_confusion_matrix
def visualize_confusion_matrix(tp_potato, fp_potato, fn_potato,
                              tp_impurity, fp_impurity, fn_impurity,
                              output_dir, title_suffix=""):
    output_vis_dir = os.path.join(output_dir, "confusion_matrix")
    os.makedirs(output_vis_dir, exist_ok=True)
    # --- FIX: Ensure inputs are integers before creating array ---
    try:
        cm = np.array([
            [int(tp_potato), int(fp_impurity)], # Row 0: True Potato (Pred P, Pred I)
            [int(fp_potato), int(tp_impurity)]  # Row 1: True Impurity(Pred P, Pred I)
        ], dtype=int) # Explicitly use integer type
    except ValueError as e:
        logger.error(f"Could not convert CM values to integers: {e}")
        logger.error(f"Received: tp_p={tp_potato}, fp_i={fp_impurity}, fp_p={fp_potato}, tp_i={tp_impurity}")
        return # Abort plotting if conversion fails

    class_names_plot = ["Potato", "Impurity"]
    try:
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_plot, yticklabels=class_names_plot) # fmt="d" should now work
        plt.xlabel('Predicted Class'); plt.ylabel('True Class'); plt.title(f'Confusion Matrix - Counts{title_suffix}'); plt.tight_layout(); cm_path = os.path.join(output_vis_dir, f"confusion_matrix_absolute{title_suffix.replace(' ', '_')}.png"); plt.savefig(cm_path); plt.close()
        logger.info(f"Saved counts confusion matrix to {cm_path}")
        cm_percent = np.zeros_like(cm, dtype=float); row_sums = cm.sum(axis=1)[:, np.newaxis]; non_zero_rows = row_sums > 0;
        if non_zero_rows.any(): cm_percent[non_zero_rows[:,0]] = cm[non_zero_rows[:,0]] / row_sums[non_zero_rows] * 100
        plt.figure(figsize=(8, 6)); sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names_plot, yticklabels=class_names_plot, vmin=0, vmax=100)
        plt.xlabel('Predicted Class'); plt.ylabel('True Class'); plt.title(f'Confusion Matrix - % (Recall Focus){title_suffix}'); plt.tight_layout(); cm_perc_path = os.path.join(output_vis_dir, f"confusion_matrix_percent{title_suffix.replace(' ', '_')}.png"); plt.savefig(cm_perc_path); plt.close()
        logger.info(f"Saved percentage confusion matrix to {cm_perc_path}")
    except Exception as e: logger.error(f"Failed confusion matrix plot generation: {e}", exc_info=True)


# ENHANCED: visualize_comparison
def visualize_comparison(image_id, original_img_path, standard_dets, optimal_dets, gt_boxes,
                         standard_thresholds, optimal_thresholds, class_names_map, output_vis_dir):
    """ Draws standard, optimal, and GT boxes, highlighting TP/FP/FN differences. """
    original_img = cv2.imread(str(original_img_path))
    if original_img is None: logger.warning(f"Could not read image: {original_img_path}"); return

    vis_img = original_img.copy()
    h, w = vis_img.shape[:2]

    # --- Colors and Styles ---
    color_std_only_tp = (0, 0, 200)       # Dark Blue (TP missed by Opt)
    color_std_only_fp = (150, 0, 0)       # Light Blue (FP unique to Std)
    color_opt_only_tp = (0, 200, 0)       # Dark Green (TP missed by Std)
    color_opt_only_fp = (0, 150, 0)       # Light Green (FP unique to Opt)
    color_common_tp = (200, 150, 150)     # Light Cyan/Pink (TP by both)
    color_common_fp = (150, 150, 200)     # Light Purple (FP by both)
    color_gt = (0, 0, 255)                # Red (Ground Truth Box)
    color_fn = (0, 165, 255)              # Orange (False Negative - GT missed by Opt)

    thickness_unique = 2; thickness_common = 1; thickness_gt = 1; thickness_fn = 1
    font_scale = 0.4; font = cv2.FONT_HERSHEY_SIMPLEX; text_color_dark_bg = (255,255,255); text_color_light_bg = (0,0,0)

    # --- Matching Logic ---
    gt_info_vis = [{'box': gt[:4], 'cls_id': int(gt[4]), 'matched_std': False, 'matched_opt': False} for gt in gt_boxes if len(gt) >= 5]
    det_info_std = [{'det': det, 'is_tp': False} for det in standard_dets]
    det_info_opt = [{'det': det, 'is_tp': False} for det in optimal_dets]

    # Match Optimal Detections to GT
    temp_gt_matched_flags = [False] * len(gt_info_vis) # Track matching within this function scope
    det_info_opt.sort(key=lambda x: x['det'][5], reverse=True) # Sort by score
    for det_item in det_info_opt:
        best_iou = 0.0; best_gt_idx = -1; det_box = det_item['det'][:4]; det_cls = int(det_item['det'][4])
        for gt_idx, gt_item in enumerate(gt_info_vis):
            if not temp_gt_matched_flags[gt_idx] and gt_item['cls_id'] == det_cls:
                iou = calculate_iou(det_box, gt_item['box'])
                if iou >= IOU_THRESHOLD and iou > best_iou: best_iou = iou; best_gt_idx = gt_idx
        if best_gt_idx != -1:
            det_item['is_tp'] = True
            gt_info_vis[best_gt_idx]['matched_opt'] = True
            temp_gt_matched_flags[best_gt_idx] = True # Mark GT as matched for optimal

    # Match Standard Detections to GT
    temp_gt_matched_flags = [False] * len(gt_info_vis) # Reset flags for standard matching
    det_info_std.sort(key=lambda x: x['det'][5], reverse=True) # Sort by score
    for det_item in det_info_std:
        best_iou = 0.0; best_gt_idx = -1; det_box = det_item['det'][:4]; det_cls = int(det_item['det'][4])
        for gt_idx, gt_item in enumerate(gt_info_vis):
             if not temp_gt_matched_flags[gt_idx] and gt_item['cls_id'] == det_cls:
                 iou = calculate_iou(det_box, gt_item['box'])
                 if iou >= IOU_THRESHOLD and iou > best_iou: best_iou = iou; best_gt_idx = gt_idx
        if best_gt_idx != -1:
             det_item['is_tp'] = True
             gt_info_vis[best_gt_idx]['matched_std'] = True
             temp_gt_matched_flags[best_gt_idx] = True # Mark GT as matched for standard

    # --- Drawing ---
    standard_sig = get_detection_signature(standard_dets); optimal_sig = get_detection_signature(optimal_dets)
    drawn_common_signatures = set()

    # 1. Draw Standard Detections
    for det_item in det_info_std:
        det = det_item['det']; is_tp = det_item['is_tp']
        x1,y1,x2,y2,cls_id,score=det; cls_id=int(cls_id); pt1,pt2=(int(x1),int(y1)),(int(x2),int(y2))
        det_sig=(cls_id,round(x1),round(y1),round(x2),round(y2));
        base_label = f"S:{class_names_map.get(cls_id,'UNK')} {score:.2f}"
        label = base_label + (" [TP]" if is_tp else " [FP]") # Add TP/FP Status

        (lbl_w, lbl_h), _ = cv2.getTextSize(label, font, font_scale, 1); text_pt = (pt1[0], pt1[1] - 2); text_bg_pt2 = (pt1[0] + lbl_w, pt1[1] - lbl_h - 3)

        if det_sig not in optimal_sig: # Standard Only
            color = color_std_only_tp if is_tp else color_std_only_fp
            cv2.rectangle(vis_img, pt1, pt2, color, thickness_unique)
            cv2.rectangle(vis_img, (pt1[0], pt1[1]), text_bg_pt2, color, -1); cv2.putText(vis_img, label, text_pt, font, font_scale, text_color_dark_bg, 1, cv2.LINE_AA)
        elif det_sig not in drawn_common_signatures: # Common
            color = color_common_tp if is_tp else color_common_fp
            cv2.rectangle(vis_img, pt1, pt2, color, thickness_common)
            cv2.rectangle(vis_img, (pt1[0], pt1[1]), text_bg_pt2, color, -1); cv2.putText(vis_img, label, text_pt, font, font_scale, text_color_light_bg, 1, cv2.LINE_AA); drawn_common_signatures.add(det_sig)

    # 2. Draw Optimal Only Detections
    for det_item in det_info_opt:
        det = det_item['det']; is_tp = det_item['is_tp']
        x1,y1,x2,y2,cls_id,score=det; cls_id=int(cls_id); pt1,pt2=(int(x1),int(y1)),(int(x2),int(y2))
        det_sig=(cls_id,round(x1),round(y1),round(x2),round(y2))
        if det_sig not in standard_sig: # Optimal Only
            base_label = f"O:{class_names_map.get(cls_id,'UNK')} {score:.2f}"
            label = base_label + (" [TP]" if is_tp else " [FP]")
            color = color_opt_only_tp if is_tp else color_opt_only_fp
            (lbl_w,lbl_h),_=cv2.getTextSize(label,font,font_scale,1); text_y=pt1[1]-lbl_h-3 if pt1[1]>(lbl_h+10) else pt2[1]+lbl_h+13; text_bg_pt1=(pt1[0],text_y-lbl_h-1); text_bg_pt2=(pt1[0]+lbl_w,text_y+1); text_pt=(pt1[0],text_y)

            cv2.rectangle(vis_img, pt1, pt2, color, thickness_unique)
            cv2.rectangle(vis_img, text_bg_pt1, text_bg_pt2, color, -1); cv2.putText(vis_img, label, text_pt, font, font_scale, text_color_light_bg, 1, cv2.LINE_AA)

    # 3. Draw Ground Truth & False Negatives (relative to Optimal)
    for gt_item in gt_info_vis:
        gt_box = gt_item['box']; cls_id = gt_item['cls_id']
        pt1, pt2 = (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3]))
        # Draw GT box always
        cv2.rectangle(vis_img, pt1, pt2, color_gt, thickness_gt)
        # If not matched by Optimal, mark as FN
        if not gt_item['matched_opt']:
            label = "FN"
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, font, font_scale, 1)
            # Position FN label near the GT box (e.g., bottom right)
            text_pt = (pt2[0] - lbl_w - 2 , pt2[1] - 2)
            cv2.putText(vis_img, label, text_pt, font, font_scale + 0.1, color_fn, 1, cv2.LINE_AA)
            # Optionally draw a dashed orange box around FN GTs for more emphasis
            # cv2.rectangle(vis_img, pt1, pt2, color_fn, thickness_fn, lineType=cv2.LINE_AA) # Example dashed line - requires drawing logic

    # --- Add Enhanced Legend ---
    legend_y = 20; line_height = 16; font_size = 0.4
    std_pc, std_ic, std_pn, std_in = standard_thresholds
    opt_pc, opt_ic, opt_pn, opt_in = optimal_thresholds
    cv2.putText(vis_img, "Legend:", (5, legend_y), font, font_size, (0,0,0), 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- TP (Std Only): Dark Blue", (10, legend_y), font, font_size, color_std_only_tp, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- FP (Std Only): Light Blue", (10, legend_y), font, font_size, color_std_only_fp, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- TP (Opt Only): Dark Green", (10, legend_y), font, font_size, color_opt_only_tp, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- FP (Opt Only): Light Green", (10, legend_y), font, font_size, color_opt_only_fp, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- TP (Common): Cyan/Pink", (10, legend_y), font, font_size, color_common_tp, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- FP (Common): Purple", (10, legend_y), font, font_size, color_common_fp, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- GT Box: Red", (10, legend_y), font, font_size, color_gt, 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"- FN (Opt Missed): Orange Text", (10, legend_y), font, font_size, color_fn, 1, cv2.LINE_AA); legend_y += line_height
    # Add threshold info
    legend_y += 5
    cv2.putText(vis_img, f"Std: Pc={std_pc:.2f},Ic={std_ic:.2f},Pn={std_pn:.2f},In={std_in:.2f}", (5, legend_y), font, font_size, (0,0,0), 1, cv2.LINE_AA); legend_y += line_height
    cv2.putText(vis_img, f"Opt: Pc={opt_pc:.3f},Ic={opt_ic:.3f},Pn={opt_pn:.3f},In={opt_in:.3f}", (5, legend_y), font, font_size, (0,0,0), 1, cv2.LINE_AA);

    save_path=os.path.join(output_vis_dir,f"{image_id}_comparison_enhanced.jpg");
    try: cv2.imwrite(save_path, vis_img)
    except Exception as e: logger.error(f"Failed save enhanced comparison: {save_path}: {e}")

# (Keep get_detection_signature, generate_analysis_plots, plot_size_bin_performance)
def get_detection_signature(detections):
    signature=set();
    for det in detections: x1,y1,x2,y2,cls_id,score=det; sig_tuple=(int(cls_id),round(x1),round(y1),round(x2),round(y2)); signature.add(sig_tuple)
    return signature
def generate_analysis_plots(results_df, output_dir, class_names_map, size_bins):
    plot_dir = os.path.join(output_dir, ANALYSIS_PLOTS_DIR); os.makedirs(plot_dir, exist_ok=True); logger.info(f"Generating analysis plots in: {plot_dir}")
    if results_df.empty: logger.warning("Results DataFrame empty. Skipping analysis plots."); return
    metrics_to_heatmap = {
        'overall_potato_precision': 'Overall Potato Precision (PMR Goal)', 'overall_impurity_recall': 'Overall Impurity Recall (IDR Goal)',
        'small_overall_f1': 'Small Object F1 (Overall, <3k)', }
    try:
        results_df['potato_conf_r']=results_df['potato_conf'].round(3); results_df['impurity_conf_r']=results_df['impurity_conf'].round(3)
        results_df['potato_nms_r']=results_df['potato_nms'].round(3); results_df['impurity_nms_r']=results_df['impurity_nms'].round(3)
        grouped = results_df.groupby(['potato_conf_r','impurity_conf_r'])
        for metric_col, title in metrics_to_heatmap.items():
             if metric_col not in results_df.columns: logger.warning(f"Metric '{metric_col}' not found for heatmap."); continue
             try:
                 avg_metric_df = grouped[metric_col].mean().reset_index(); pivot_df = avg_metric_df.pivot_table(index='impurity_conf_r', columns='potato_conf_r', values=metric_col)
                 plt.figure(figsize=(10,8)); sns.heatmap(pivot_df.sort_index(ascending=False),annot=True,fmt=".3f",cmap="viridis",linewidths=.5,cbar_kws={'label':f"Avg. {title}"})
                 plt.xlabel("Potato Confidence Threshold"); plt.ylabel("Impurity Confidence Threshold"); plt.title(f"Heatmap vs Conf (Avg over NMS): {title}"); plt.tight_layout()
                 plot_path=os.path.join(plot_dir,f"heatmap_avgNMS_{metric_col}.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved heatmap (Avg NMS): {plot_path}")
             except Exception as e: logger.error(f"Failed heatmap for {metric_col}: {e}", exc_info=False)
    except Exception as e: logger.error(f"Error heatmap prep: {e}", exc_info=True)
    try:
        plt.figure(figsize=(10,8)); scatter=plt.scatter(results_df['overall_impurity_recall'],results_df['overall_potato_precision'],c=results_df['small_overall_f1'],cmap='viridis',alpha=0.6,s=30)
        plt.colorbar(scatter,label='Small Object F1 (Overall, <3k)'); plt.axhline(TARGET_POTATO_PRECISION,color='red',linestyle='--',linewidth=1,label=f'Target Potato P = {TARGET_POTATO_PRECISION}')
        plt.axvline(TARGET_IMPURITY_RECALL,color='blue',linestyle='--',linewidth=1,label=f'Target Impurity R = {TARGET_IMPURITY_RECALL}')
        plt.fill_betweenx(y=[TARGET_POTATO_PRECISION, 1.01], x1=TARGET_IMPURITY_RECALL, x2=1.01, color='green', alpha=0.1, label='Target Zone')
        plt.xlabel("Overall Impurity Recall (IDR)"); plt.ylabel("Overall Potato Precision (1-PMR)"); plt.title("Performance Trade-off (Overall Metrics)")
        min_r=max(0,results_df['overall_impurity_recall'].min()-0.05); min_p=max(0.9,results_df['overall_potato_precision'].min()-0.01)
        plt.xlim(left=min_r, right=1.01); plt.ylim(bottom=min_p, top=1.005); plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='lower right'); plt.tight_layout()
        plot_path=os.path.join(plot_dir,"pr_scatter_overall_goals.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved P-R scatter plot (Overall Goals): {plot_path}")
    except Exception as e: logger.error(f"Failed P-R scatter plot: {e}", exc_info=True)

def plot_size_bin_performance(best_result_dict, class_names_map, size_bins, output_dir):
    plot_dir = os.path.join(output_dir, ANALYSIS_PLOTS_DIR); logger.info("Generating size bin performance plot...");
    if not best_result_dict: logger.warning("No best result data to plot size bin performance."); return
    data = []
    for cls_id, cls_name in class_names_map.items():
        safe_cls_name = "".join(c if c.isalnum() else "_" for c in cls_name)
        for bin_idx, size_bin in enumerate(size_bins):
            size_label = get_size_bin_label(size_bin[0], size_bins)
            safe_size_label = size_label.replace('<','lt').replace('>=','gte').replace('-','_').replace(' ','')
            prefix = f'class_{safe_cls_name}_size_{safe_size_label}_'
            if f'{prefix}f1' in best_result_dict:
                data.append({'Class':cls_name,'Size Bin':size_label,'Precision':best_result_dict.get(f'{prefix}precision',0.0),'Recall':best_result_dict.get(f'{prefix}recall',0.0),'F1':best_result_dict.get(f'{prefix}f1',0.0),'TP':best_result_dict.get(f'{prefix}tp',0),'Bin Index':bin_idx})
    if not data: logger.warning("No valid size bin data found in best result."); return
    df = pd.DataFrame(data); df = df.sort_values(by=['Class', 'Bin Index'])
    try:            
        plt.figure(figsize=(14,8)); sns.barplot(data=df,x='Size Bin',y='F1',hue='Class',palette='viridis')
        plt.xlabel("Object Size Bin (pixels²)"); plt.ylabel("F1-Score"); plt.title(f"F1-Score by Object Size and Class (Optimal Thresholds)"); plt.xticks(rotation=45, ha='right'); plt.ylim(0,1.05); plt.grid(axis='y',linestyle=':',alpha=0.7); plt.legend(title='Class',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout()
        plot_path = os.path.join(plot_dir, "bar_size_bin_f1_optimal.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved size bin F1 plot: {plot_path}")
    except Exception as e: logger.error(f"Failed size bin plot: {e}", exc_info=True)


# --- Logging Setup ---
def setup_logging(output_dir):
    logger.remove(); log_format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>|<level>{level:<8}</level>|<level>{message}</level>"; logger.add(sys.stderr,level="INFO",format=log_format)
    log_file_path=os.path.join(output_dir,"evaluation_log_v5_1_trt.log"); logger.add(log_file_path,level="DEBUG",format=log_format,rotation="10 MB")

# --- JSON Serialization Helper ---
# (Keep make_serializable from previous)
def make_serializable(data):
    if isinstance(data,dict): return {k:make_serializable(v) for k,v in data.items()}
    if isinstance(data,list): return [make_serializable(i) for i in data]
    if isinstance(data,(np.int_,np.intc,np.intp,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)): return int(data)
    if isinstance(data,(np.float_,np.float16,np.float32,np.float64,float)):
        if np.isnan(data): return None;
        if np.isinf(data): return str(data)
        return float(data)
    if isinstance(data,(np.ndarray,)): return make_serializable(data.tolist())
    if isinstance(data,np.bool_): return bool(data)
    return data

# --- mAP Calculation Placeholder ---
# (Keep calculate_map from previous)
def calculate_map(model_path_or_obj, dataset_yaml, device):
    logger.info(f"Calculating mAP@0.5 for {model_path_or_obj} (may take time)...")
    try:
        model = YOLO(model_path_or_obj)
        metrics = model.val(data=dataset_yaml,iou=0.5,conf=0.001,split='test',device=device,plots=False,verbose=False)
        map50 = metrics.box.map50
        logger.info(f"Ultralytics Validation Results: mAP@0.5 = {map50:.4f}")
        return map50
    except Exception as e: logger.error(f"Failed mAP calculation: {e}"); return -1.0

# --- Main Execution Logic ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)
    logger.info("Starting YOLOv12 Comprehensive Evaluation v5.1 (TensorRT ENGINE ONLY)")
    # ... [Parameter Logging remains same] ...
    raw_data_filepath=os.path.join(OUTPUT_DIR,RAW_DATA_FILENAME); results_filepath=os.path.join(OUTPUT_DIR,RESULTS_FILENAME); best_result_filepath=os.path.join(OUTPUT_DIR,BEST_RESULT_FILENAME); comparison_vis_dir=os.path.join(OUTPUT_DIR,COMPARISON_VIS_DIR); analysis_plots_dir=os.path.join(OUTPUT_DIR,ANALYSIS_PLOTS_DIR)
    # --- Load Dataset Info ---
    # (Dataset loading remains same)
    try:
        with open(DATASET_YAML,'r',encoding='utf-8') as f: dataset_info=yaml.safe_load(f); class_names=dataset_info.get('names',[])
        if not class_names: logger.error(f"No names in {DATASET_YAML}"); sys.exit(1)
        logger.info(f"Classes: {class_names}"); class_names_map={i:name for i,name in enumerate(class_names)}
        all_defined_ids=POTATO_CLS_IDS.union(IMPURITY_CLS_IDS)
        if not all(0<=i<len(class_names) for i in all_defined_ids): logger.error(f"Class IDs invalid {all_defined_ids} for {class_names}"); sys.exit(1)
        if POTATO_CLS_IDS.intersection(IMPURITY_CLS_IDS): logger.error("Class IDs overlap!"); sys.exit(1)
        yaml_dir=Path(DATASET_YAML).parent; img_path_key='test' if 'test' in dataset_info else ('val' if 'val' in dataset_info else None)
        if not img_path_key: logger.error("No 'test'/'val' in YAML"); sys.exit(1)
        img_path_rel=dataset_info[img_path_key]; image_dir_path=(yaml_dir/img_path_rel).resolve() if not Path(img_path_rel).is_absolute() else Path(img_path_rel)
        logger.info(f"Images: '{img_path_key}' set: {image_dir_path}");
        if not image_dir_path.is_dir(): logger.error(f"Not found: {image_dir_path}"); sys.exit(1)
        label_dir_path=(image_dir_path.parent.parent/'labels'/image_dir_path.name).resolve()
        if not label_dir_path.is_dir(): label_dir_path=(image_dir_path.parent/'labels').resolve()
        logger.info(f"Labels: {label_dir_path}");
        if not label_dir_path.is_dir(): logger.warning(f"Not found: {label_dir_path}. GT loading will fail.")
    except Exception as e: logger.error(f"Dataset config error {DATASET_YAML}: {e}"); sys.exit(1)
    num_classes=len(class_names)
    # --- mAP Calculation Call (Optional) ---
    # map50_result = calculate_map(MODEL_PATH, DATASET_YAML, DEVICE)

    # --- Prepare Data (Load/Generate Raw) ---
    # (Data preparation logic remains same)
    all_raw_detections={}; all_ground_truths={}; inference_times=[]
    if os.path.exists(raw_data_filepath):
        logger.info(f"Loading raw data from {raw_data_filepath}...")
        loaded_data=load_raw_data(raw_data_filepath)
        if loaded_data:
             all_raw_detections=loaded_data.get("detections",{}); all_ground_truths=loaded_data.get("ground_truths",{})
             loaded_class_names=loaded_data.get("class_names",[]);
             if not all_raw_detections or not all_ground_truths: logger.error("Incomplete raw data."); sys.exit(1)
             if loaded_class_names!=class_names: logger.warning(f"Class name mismatch! Using YAML: {class_names}")
             logger.info(f"Loaded raw data for {len(all_raw_detections)} images.")
        else: logger.error("Failed load raw data."); sys.exit(1)
    else:
        logger.info("Generating raw data via TRT engine inference...")
        if not os.path.exists(MODEL_PATH) or not MODEL_PATH.endswith(".engine"): logger.error(f"TRT model (.engine) not found/invalid: {MODEL_PATH}"); sys.exit(1)
        try: model=YOLO(MODEL_PATH); logger.info(f"Loaded TRT engine: {MODEL_PATH}")
        except Exception as e: logger.error(f"Failed load TRT engine {MODEL_PATH}: {e}"); sys.exit(1)
        image_files=sorted([p for p in image_dir_path.glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']])
        if not image_files: logger.error(f"No images found: {image_dir_path}"); sys.exit(1)
        logger.info(f"Found {len(image_files)} images.")
        CONF_THRESH_FOR_RAW=0.001; IOU_THRESH_FOR_RAW=0.7
        for img_path in tqdm(image_files, desc="Collecting Raw Data (TRT)"):
             image_id=img_path.stem; label_path=label_dir_path/f"{image_id}.txt"
             try:
                img=cv2.imread(str(img_path));
                if img is None: logger.warning(f"Skip unloadable: {img_path}"); continue
                h,w=img.shape[:2]; gt_boxes=load_yolo_annotation(str(label_path),w,h,class_names); all_ground_truths[image_id]=gt_boxes
                inf_start=time.time(); results=model.predict(source=img,conf=CONF_THRESH_FOR_RAW,iou=IOU_THRESH_FOR_RAW,device=DEVICE,verbose=False)
                inf_end=time.time(); inference_times.append(inf_end-inf_start); raw_dets_img=[]
                if results and results[0].boxes:
                    boxes=results[0].boxes.xyxy.cpu().numpy(); confs=results[0].boxes.conf.cpu().numpy(); cls_ids=results[0].boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(boxes)):
                         if 0<=cls_ids[i]<num_classes and confs[i]>=CONF_THRESH_FOR_RAW: raw_dets_img.append([*boxes[i],cls_ids[i],confs[i]])
                all_raw_detections[image_id]={'detections':raw_dets_img,'dims':(w,h)}
             except Exception as e: logger.error(f"Error processing {img_path}: {e}",exc_info=False)
        data_to_save={"detections":all_raw_detections,"ground_truths":all_ground_truths,"class_names":class_names}; save_raw_data(data_to_save,raw_data_filepath)
        if inference_times: avg_inf_ms=(sum(inference_times)/len(inference_times))*1000; fps=1.0/(avg_inf_ms/1000) if avg_inf_ms>0 else 0; logger.info(f"Raw data collection (TRT). Avg inference: {avg_inf_ms:.2f} ms ({fps:.2f} FPS)")

    # --- Offline Analysis Phase ---
    # (Analysis, Saving, Plotting logic remains same)
    if not all_raw_detections or not all_ground_truths: logger.error("Missing data for analysis."); sys.exit(1)
    logger.info("Starting comprehensive offline analysis with NMS tuning...")
    combinations=[(round(pc,3),round(ic,3),round(pn,3),round(imn,3)) for pc in POTATO_CONF_RANGE for ic in IMPURITY_CONF_RANGE for pn in POTATO_NMS_RANGE for imn in IMPURITY_NMS_RANGE]
    logger.info(f"Total combinations: {len(combinations)}")
    if not combinations: logger.error("No combinations generated."); sys.exit(1)
    all_results=[]
    try:
        num_workers=os.cpu_count(); logger.info(f"Starting parallel evaluation ({num_workers} workers)...")
        worker_func=functools.partial(evaluate_combination,all_raw_detections=all_raw_detections,all_ground_truths=all_ground_truths,potato_cls_ids=POTATO_CLS_IDS,impurity_cls_ids=IMPURITY_CLS_IDS,num_classes=num_classes,iou_threshold=IOU_THRESHOLD,size_bins=SIZE_BINS,class_names_map=class_names_map)
        with multiprocessing.Pool(processes=num_workers) as pool:
             results_iterator=pool.imap_unordered(worker_func,combinations)
             for result in tqdm(results_iterator,total=len(combinations),desc="Analyzing Thresholds"):
                 if result is not None: all_results.append(result)
        logger.info(f"Finished parallel evaluation. Collected {len(all_results)} results.")
    except KeyboardInterrupt: logger.warning("Analysis interrupted."); sys.exit(1)
    except Exception as pool_e: logger.error(f"Multiprocessing failed: {pool_e}"); sys.exit(1)
    if not all_results: logger.error("No results collected."); sys.exit(1)
    logger.info(f"Saving all {len(all_results)} results to {results_filepath}...")
    try:
        results_df=pd.DataFrame(all_results); results_df.sort_values(by=['potato_conf','impurity_conf','potato_nms','impurity_nms'],inplace=True)
        for col in results_df.select_dtypes(include=np.number).columns: results_df[col]=pd.to_numeric(results_df[col],errors='coerce')
        results_df.to_csv(results_filepath,index=False,float_format='%.6f'); logger.info("Saved all results to CSV.")
        generate_analysis_plots(results_df,OUTPUT_DIR,class_names_map,SIZE_BINS)
    except Exception as e:
        logger.error(f"Failed save/plot results: {e}",exc_info=True);
        if 'results_df' not in locals(): results_df=pd.DataFrame(all_results)
    logger.info("Selecting best threshold based on OVERALL goals...")
    if results_df.empty: logger.error("Results DF empty for selection."); sys.exit(1)
    candidates_df=results_df[(results_df['overall_potato_precision']>=TARGET_POTATO_PRECISION)&(results_df['overall_impurity_recall']>=TARGET_IMPURITY_RECALL)].copy()
    logger.info(f"Found {len(candidates_df)} combinations meeting OVERALL goals.")
    best_result_dict=None
    if not candidates_df.empty:
        candidates_df=candidates_df.sort_values(by=['small_overall_f1','overall_impurity_recall'],ascending=[False,False])
        best_result_dict=candidates_df.iloc[0].to_dict(); logger.info(f"Selected best: Conf_P={best_result_dict['potato_conf']:.3f}, Conf_I={best_result_dict['impurity_conf']:.3f}, NMS_P={best_result_dict['potato_nms']:.3f}, NMS_I={best_result_dict['impurity_nms']:.3f}")
    else: logger.warning("No combination met OVERALL goals!")
    
    if best_result_dict:
        logger.info("--- Final Optimal Result (Overall Goals & Small F1) ---")
        pc=best_result_dict['potato_conf'];ic=best_result_dict['impurity_conf'];pn=best_result_dict['potato_nms'];imn=best_result_dict['impurity_nms']
        logger.info(f"Thresholds: Pc={pc:.3f}, Ic={ic:.3f}, Pn={pn:.3f}, In={imn:.3f}")
    
        logger.info(" Metrics (Overall):")
        logger.info(f"  Potato Precision: {best_result_dict.get('overall_potato_precision',-1):.6f} (Target >= {TARGET_POTATO_PRECISION})")
        logger.info(f"  Impurity Recall:  {best_result_dict.get('overall_impurity_recall',-1):.6f} (Target >= {TARGET_IMPURITY_RECALL})")
        logger.info(f"  Small Obj F1 (<3k):{best_result_dict.get('small_overall_f1',-1):.6f}")
        logger.info(f"  Potato Recall:    {best_result_dict.get('overall_potato_recall',-1):.6f}")
        logger.info(f"  Impurity Precision:{best_result_dict.get('overall_impurity_precision',-1):.6f}")
    
        logger.info(" Metrics (Per Class - Overall Size):")
        for cls_id in range(num_classes):
             cls_name=class_names_map.get(cls_id,f"Class_{cls_id}"); safe_cls_name="".join(c if c.isalnum() else "_" for c in cls_name)
             p_key,r_key,f1_key=f"class_{safe_cls_name}_precision",f"class_{safe_cls_name}_recall",f"class_{safe_cls_name}_f1"
             if p_key in best_result_dict: logger.info(f"  {cls_name}: P={best_result_dict[p_key]:.4f}, R={best_result_dict[r_key]:.4f}, F1={best_result_dict[f1_key]:.4f}")
    
        # Output: Object size per class metrics
        logger.info(" Metrics (Per Class - Per Size Bin):")
        for cls_id in range(num_classes):
            cls_name=class_names_map.get(cls_id,f"Class_{cls_id}")
            safe_cls_name="".join(c if c.isalnum() else "_" for c in cls_name)
            logger.info(f"  Class: {cls_name}")
        
            for bin_idx, size_bin in enumerate(size_bins):
                
                size_label = get_size_bin_label(size_bin[0], size_bins)
                safe_size_label = size_label.replace('<','lt').replace('>=','gte').replace('-','_').replace(' ','')
                prefix = f'class_{safe_cls_name}_size_{safe_size_label}_'
            
                if f'{prefix}f1' in best_result_dict:
                    p_val = best_result_dict.get(f'{prefix}precision', -1)
                    r_val = best_result_dict.get(f'{prefix}recall', -1)
                    f1_val = best_result_dict.get(f'{prefix}f1', -1)
                    tp_val = best_result_dict.get(f'{prefix}tp', 0)
                    fp_val = best_result_dict.get(f'{prefix}fp', 0)
                    fn_val = best_result_dict.get(f'{prefix}fn', 0)
                
                    if tp_val > 0 or fp_val > 0 or fn_val > 0:  
                        logger.info(f"    Size {size_label}: P={p_val:.4f}, R={r_val:.4f}, F1={f1_val:.4f} (TP={tp_val}, FP={fp_val}, FN={fn_val})")
    
        try:
             logger.info(f"Saving best result to {best_result_filepath}...");     serializable_best=make_serializable(best_result_dict);
             with open(best_result_filepath,'w') as f: json.dump(serializable_best,f,indent=4); logger.info("Best result saved.")
        except Exception as e: logger.error(f"Failed save best result JSON: {e}")
        
        visualize_confusion_matrix(tp_potato=best_result_dict.get('overall_potato_tp',0),fp_potato=best_result_dict.get('overall_impurity_fp',0),fn_potato=best_result_dict.get('overall_potato_fn',0),tp_impurity=best_result_dict.get('overall_impurity_tp',0),fp_impurity=best_result_dict.get('overall_potato_fp',0),fn_impurity=best_result_dict.get('overall_impurity_fn',0),output_dir=OUTPUT_DIR,title_suffix=" (Optimal, Overall)")
        plot_size_bin_performance(best_result_dict, class_names_map, SIZE_BINS, OUTPUT_DIR)
        logger.info("Generating enhanced comparison visualizations...")
        os.makedirs(comparison_vis_dir, exist_ok=True); optimal_thresholds_tuple=(pc,ic,pn,imn); standard_thresholds_tuple=(STANDARD_POTATO_CONF,STANDARD_IMPURITY_CONF,STANDARD_POTATO_NMS,STANDARD_IMPURITY_NMS)
        vis_count=0; processed_count=0
        try: image_files_for_vis={p.stem: p for p in image_dir_path.glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']}
        except Exception as e: logger.error(f"Could not list images: {e}"); image_files_for_vis={}
        if image_files_for_vis:
             for image_id in tqdm(all_raw_detections.keys(),desc="Generating Enhanced Comparison Images"):
                 processed_count+=1;
                 if image_id not in image_files_for_vis: continue
                 try:
                     raw_data=all_raw_detections[image_id]; gt_boxes=all_ground_truths.get(image_id,np.array([])); raw_dets=raw_data.get('detections',[])
                     standard_dets=apply_confidence_and_nms(raw_dets,standard_thresholds_tuple[0],standard_thresholds_tuple[1],standard_thresholds_tuple[2],standard_thresholds_tuple[3],POTATO_CLS_IDS,IMPURITY_CLS_IDS)
                     optimal_dets=apply_confidence_and_nms(raw_dets,optimal_thresholds_tuple[0],optimal_thresholds_tuple[1],optimal_thresholds_tuple[2],optimal_thresholds_tuple[3],POTATO_CLS_IDS,IMPURITY_CLS_IDS)
                     standard_sig=get_detection_signature(standard_dets); optimal_sig=get_detection_signature(optimal_dets)
                     if standard_sig!=optimal_sig: vis_count+=1; visualize_comparison(image_id,image_files_for_vis[image_id],standard_dets,optimal_dets,gt_boxes,standard_thresholds_tuple,optimal_thresholds_tuple,class_names_map,comparison_vis_dir)
                 except Exception as e: logger.error(f"Error enhanced comparison {image_id}: {e}",exc_info=False)
             logger.info(f"Enhanced comparison visualization done. Saved {vis_count}/{processed_count} images.")
        else: logger.warning("Image dir unreadable. Skipping enhanced comparison vis.")
    else: logger.error("Could not determine optimal combination.")
    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True); logger.info("Set multiprocessing start method 'spawn'")
    except RuntimeError as e: logger.warning(f"Start method set failed: {e}. Using default '{multiprocessing.get_start_method()}'.")
    except Exception as e: logger.warning(f"Error setting start method: {e}")
    main()
