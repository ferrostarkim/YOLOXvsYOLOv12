# -*- coding:utf-8 -*-
# YOLOX Comprehensive Evaluation Script (TensorRT ENGINE Primarily - v2-trt)
# Based on Model_test_multi.py, adapted for comprehensive metrics like yolov12_evaluation_v5_trt.py

import argparse
import os
import time
import sys
import cv2
import numpy as np
import torch # Still needed for tensor operations and TRT module interaction
import xml.etree.ElementTree as ET
import json
import pickle
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing
import functools
from pathlib import Path # Added for path handling consistency

# --- Configuration (Mirrors YOLOv12 script where applicable) ---

# <<<--- Important: This script focuses on YOLOX TensorRT engine evaluation. ---<<<
# <<<--- PyTorch(.pth) can also be loaded, but TRT is recommended. ---<<<

# Output Paths
OUTPUT_DIR = 'yolox_trt_eval_results_v2_comprehensive' # Changed output dir name
RAW_DATA_FILENAME = "raw_eval_data_yolox.pkl" # Specific filename for YOLOX raw data
RESULTS_FILENAME = "all_evaluation_results_yolox_v2.csv" # Changed results filename
BEST_RESULT_FILENAME = "best_evaluation_result_yolox_v2.json" # Filename for best result
COMPARISON_VIS_DIR = "comparison_visuals_yolox_enhanced" # Subdir for enhanced comparison images
ANALYSIS_PLOTS_DIR = "analysis_plots_yolox" # Subdir for analysis plots

# Class Names (Consistent with original script and assumed dataset)
CLASS_NAMES = ["baresho", "dokai", "ishi"] # Potato, Clod, Stone
POTATO_CLS_NAME = "baresho"
IMPURITY_CLS_NAMES = {"dokai", "ishi"}

# Evaluation Parameters
IOU_THRESHOLD = 0.5 # IoU threshold for TP/FP matching

# Size Bins Definition (Mirrors YOLOv12 script)
SIZE_BINS = [
    (0, 3000),          # Small (< 3k)
    (3000, 6000),       # 3k-6k
    (6000, 9000),       # 6k-9k
    (9000, 12000),      # 9k-12k
    (12000, 15000),     # 12k-15k
    (15000, 18000),     # 15k-18k
    (18000, 21000),     # 18k-21k
    (21000, 24000),     # 21k-24k
    (24000, float('inf')) # Large (>= 24k)
]
def get_size_bin_label(area, bins): # Helper function for labels
    for lower, upper in bins:
        if lower <= area < upper:
            # Convert to safe label format used in column names
            if upper == float('inf'): return f"gte{int(lower/1000)}k"
            return f"{int(lower/1000)}k_{int(upper/1000)}k"
            # Original label format (for plotting):
            # if upper == float('inf'): return f">={int(lower/1000)}k"
            # return f"{int(lower/1000)}k-{int(upper/1000)}k"
    return "Unknown"

DEVICE = "cuda:0" # Default device, consider making GPU index explicit

# Standard Thresholds for Comparison Visualization (Mirrors YOLOv12 script)
STANDARD_POTATO_CONF = 0.5
STANDARD_IMPURITY_CONF = 0.5
STANDARD_POTATO_NMS = 0.45
STANDARD_IMPURITY_NMS = 0.45

# Threshold Ranges for Analysis (Mirrors YOLOv12 script)
POTATO_CONF_RANGE = np.round(np.arange(0.45, 0.86, 0.1), 3)
IMPURITY_CONF_RANGE = np.round(np.arange(0.2, 0.66, 0.1), 3)
POTATO_NMS_RANGE = np.round(np.arange(0.35, 0.56, 0.1), 3)
IMPURITY_NMS_RANGE = np.round(np.arange(0.35, 0.56, 0.1), 3)

# System Goal Targets (Applied to OVERALL metrics - Mirrors YOLOv12 script)
TARGET_POTATO_PRECISION = 0.990 # Equivalent to PMR < 1%
TARGET_IMPURITY_RECALL = 0.600  # IDR >= 60%

# --- YOLOX specific imports ---
try:
    from yolox.data.data_augment import ValTransform
    from yolox.exp import get_exp
    # Removed fuse_model import as TRT is preferred
    from yolox.utils import get_model_info, postprocess
except ImportError:
    logger.error("YOLOX components not found. Ensure YOLOX is installed correctly.")
    sys.exit(1)

# --- torch2trt import ---
try:
    from torch2trt import TRTModule
except ImportError:
    TRTModule = None
    logger.warning("torch2trt not found. --trt flag will not work.")

# --- Utility Functions (calculate_iou, get_box_area - NEW) ---
def calculate_iou(box1, box2):
    # (Keep the robust calculate_iou from original script)
    if not (len(box1) >= 4 and len(box2) >= 4): return 0.0
    if box1[2] <= box1[0] or box1[3] <= box1[1] or box2[2] <= box2[0] or box2[3] <= box2[1]: return 0.0
    x1_inter = max(box1[0], box2[0]); y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2]); y2_inter = min(box1[3], box2[3])
    width_inter = max(0, x2_inter - x1_inter); height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area_box1 <= 0 or area_box2 <= 0: return 0.0
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union if area_union > 0 else 0
    return float(iou)

def get_box_area(box): # Added for consistency with YOLOv12 script
    if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]: return 0.0
    return float((box[2] - box[0]) * (box[3] - box[1]))

# --- Data Loading (VOC XML) ---
class VOCAnnotationLoader:
    # (Keep the robust VOCAnnotationLoader from original script)
    def __init__(self, annotations_dir, class_names):
        self.annotations_dir = annotations_dir
        self.class_names = class_names
        self.class_dict = {name: idx for idx, name in enumerate(class_names)}
        logger.info(f"Annotation class mapping: {self.class_dict}")
    def get_annotation(self, image_id):
        xml_file = os.path.join(self.annotations_dir, image_id + ".xml")
        if not os.path.exists(xml_file): return np.array([])
        try: tree = ET.parse(xml_file); root = tree.getroot()
        except ET.ParseError: logger.error(f"Failed XML parse: {xml_file}"); return np.array([])
        boxes = []; img_width, img_height = 0, 0
        size_elem = root.find('size');
        if size_elem is not None:
            try: img_width = float(size_elem.find('width').text); img_height = float(size_elem.find('height').text)
            except (ValueError, TypeError): img_width, img_height = 0, 0
        for obj in root.findall('object'):
            try:
                class_name = obj.find('name').text;
                if class_name not in self.class_dict: continue
                bbox = obj.find('bndbox'); x1 = float(bbox.find('xmin').text); y1 = float(bbox.find('ymin').text); x2 = float(bbox.find('xmax').text); y2 = float(bbox.find('ymax').text)
                if x1 >= x2 or y1 >= y2: continue
                if img_width > 0 and img_height > 0: x1=max(0.0,x1); y1=max(0.0,y1); x2=min(img_width,x2); y2=min(img_height,y2); if x1>=x2 or y1>=y2: continue
                if (x2 - x1) < 1 or (y2 - y1) < 1: continue # Min size check
                class_id = self.class_dict[class_name]
                boxes.append([x1, y1, x2, y2, class_id]) # Store with class ID
            except (AttributeError, ValueError, TypeError) as e: continue
        return np.array(boxes, dtype=np.float32)

# --- Model Inference (TRT Focused) ---
class TRTPredictor:
    # (Keep TRTPredictor from original script, ensure TRT loading is primary path)
    def __init__(self, model, exp, cls_names, trt_file=None, decoder=None,
                 device="gpu", fp16=False, legacy=False):
        self.model = model # PyTorch model initially
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = 0.001 # Use very low conf for raw data collection
        self.nmsthre = exp.nmsthre # Use default NMS from experiment file for raw data
        self.test_size = exp.test_size
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu") # Be explicit
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.exp = exp

        if trt_file is not None:
            if TRTModule is None: raise ImportError("torch2trt required for --trt")
            logger.info(f"Loading TRT model from: {trt_file}")
            try:
                model_trt = TRTModule(); map_location = self.device
                model_trt.load_state_dict(torch.load(trt_file, map_location=map_location))
                self.model = model_trt # Use TRT model
                self.model.to(self.device).eval()
                logger.info("Predictor using loaded TensorRT model.")
                # Simplified warmup (optional but good practice)
                try:
                    logger.info("Performing TRT warmup...")
                    dummy_input = torch.ones(1, 3, *self.test_size, device=self.device)
                    if self.fp16: dummy_input = dummy_input.half()
                    with torch.no_grad(): _ = self.model(dummy_input)
                    logger.info("TRT warmup finished.")
                except Exception as wu_e: logger.warning(f"TRT warmup failed: {wu_e}")
            except Exception as e: logger.error(f"TRT model load/setup failed: {e}", exc_info=True); raise e
        else:
            # If not using TRT, load PyTorch checkpoint (logic moved to main)
            logger.info("Predictor using PyTorch model.")
            self.model.to(self.device)
            if self.fp16: self.model.half()
            self.model.eval()

    def inference(self, img):
        # (Keep inference logic from original, ensures raw output before NMS/Conf filtering)
        img_info = {"id": 0}
        if isinstance(img,str): img_info["file_name"]=os.path.basename(img); img=cv2.imread(img)
        else: img_info["file_name"]=None
        if img is None: logger.error("Input image is None."); return None,img_info
        height,width=img.shape[:2]; img_info["height"]=height; img_info["width"]=width
        ratio = min(self.test_size[0]/height, self.test_size[1]/width); img_info["ratio"]=ratio
        img_tensor,_ = self.preproc(img,None,self.test_size); img_tensor=torch.from_numpy(img_tensor).unsqueeze(0).float().to(self.device)
        if self.fp16: img_tensor=img_tensor.half()
        start_time = time.time(); raw_model_outputs = None
        with torch.no_grad():
            try: raw_model_outputs=self.model(img_tensor)
            except Exception as e: logger.error(f"Inference failed: {e}"); return None,img_info
        postprocessed_output_list = None
        try:
            processed_for_postprocess = raw_model_outputs
            if self.decoder and isinstance(self.model, TRTModule):
                 # Optional decoder logic here if needed for specific YOLOX TRT models
                 logger.warning("Decoder logic needs verification for YOLOX TRT outputs.")
                 if isinstance(processed_for_postprocess, torch.Tensor):
                      processed_for_postprocess = self.decoder(processed_for_postprocess, dtype=processed_for_postprocess.type())

            # Postprocess using the low confidence threshold set in __init__
            postprocessed_output_list = postprocess(processed_for_postprocess, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
        except Exception as e: logger.error(f"Postprocessing failed: {e}"); return None,img_info
        # Extract the raw tensor result: [N, 7] -> [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        raw_dets_tensor_for_saving = postprocessed_output_list[0] if postprocessed_output_list and postprocessed_output_list[0] is not None else None
        end_time = time.time(); img_info["inference_time"] = end_time - start_time
        return raw_dets_tensor_for_saving, img_info

# --- Data Saving/Loading ---
# (Keep save_raw_data, load_raw_data from original script)
def save_raw_data(data, filepath):
    logger.info(f"Saving raw data to {filepath}...");
    try:
        with open(filepath, 'wb') as f: pickle.dump(data, f); logger.info("Raw data saved.")
    except Exception as e: logger.error(f"Save raw data failed: {e}")
def load_raw_data(filepath):
    logger.info(f"Loading raw data from {filepath}...");
    if not os.path.exists(filepath): logger.error(f"Raw data file not found: {filepath}"); return None
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f); logger.info("Raw data loaded."); return data
    except Exception as e: logger.error(f"Failed load raw data: {e}"); return None

# --- Offline Analysis Core Functions ---

# NMS Function (Same as YOLOv12 script)
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

# Combined Confidence and NMS Filtering (Adapted for YOLOX structure)
def apply_confidence_and_nms_yolox(raw_detections_tensor, # Takes tensor output from YOLOX postprocess
                                  potato_conf_thresh, impurity_conf_thresh,
                                  potato_nms_thresh, impurity_nms_thresh,
                                  class_names, potato_cls_name, impurity_cls_names):
    """ Applies confidence and NMS filtering to YOLOX raw postprocessed tensor. """
    if raw_detections_tensor is None: return []
    detections_np = raw_detections_tensor.cpu().numpy()
    class_id_map = {name: i for i, name in enumerate(class_names)}
    potato_cls_id = class_id_map.get(potato_cls_name, -1)
    impurity_cls_ids = {class_id_map.get(name, -1) for name in impurity_cls_names if name in class_id_map}

    conf_filtered_potato = []; conf_filtered_impurity = []
    for i in range(len(detections_np)):
        box = detections_np[i, :4].tolist(); obj_conf = float(detections_np[i, 4]); cls_conf = float(detections_np[i, 5]); cls_id = int(detections_np[i, 6])
        score = obj_conf * cls_conf # YOLOX score calculation
        if cls_id == potato_cls_id:
            if score >= potato_conf_thresh: conf_filtered_potato.append(box + [cls_id, score])
        elif cls_id in impurity_cls_ids:
            if score >= impurity_conf_thresh: conf_filtered_impurity.append(box + [cls_id, score])
    nms_filtered_potato = apply_nms_to_detections(conf_filtered_potato, potato_nms_thresh)
    nms_filtered_impurity = apply_nms_to_detections(conf_filtered_impurity, impurity_nms_thresh)
    final_detections = nms_filtered_potato + nms_filtered_impurity
    return final_detections

# Comprehensive Evaluation Logic (Adapted from YOLOv12 script)
def evaluate_precision_recall_comprehensive_yolox(detections, ground_truths, num_classes, iou_threshold,
                                                  class_names, potato_cls_name, impurity_cls_names, size_bins):
    # (Same logic as before, calculates detailed TP/FP/FN per class/size bin)
    stats = {
        'overall_potato': {'tp': 0,'fp': 0,'fn': 0}, 'overall_impurity': {'tp': 0,'fp': 0,'fn': 0},
        'per_class': defaultdict(lambda: {'tp': 0,'fp': 0,'fn': 0}),
        'per_size_bin': defaultdict(lambda: {'tp': 0,'fp': 0,'fn': 0}),
        'per_class_size_bin': defaultdict(lambda: defaultdict(lambda: {'tp': 0,'fp': 0,'fn': 0}))
    }
    class_id_map = {name: i for i, name in enumerate(class_names)}
    potato_cls_id = class_id_map.get(potato_cls_name, -1)
    impurity_cls_ids = {class_id_map.get(name, -1) for name in impurity_cls_names if name in class_id_map}
    size_bin_labels = {i: get_size_bin_label(0, size_bins) for i in range(len(size_bins))}
    gt_boxes_list = ground_truths.tolist() if isinstance(ground_truths, np.ndarray) else ground_truths
    det_boxes_list = detections
    if not gt_boxes_list and not det_boxes_list: return stats
    gt_info = []
    for i, gt in enumerate(gt_boxes_list):
        if len(gt) < 5: continue
        gt_box=gt[:4]; gt_cls=int(gt[4]); gt_area=get_box_area(gt_box); is_potato=gt_cls == potato_cls_id; is_impurity=gt_cls in impurity_cls_ids; size_bin_idx=-1
        for idx, (lower, upper) in enumerate(size_bins):
             if lower <= gt_area < upper: size_bin_idx = idx; break
        gt_info.append({'index': i,'box': gt_box,'class': gt_cls,'area': gt_area,'is_potato': is_potato,'is_impurity': is_impurity,'size_bin_idx': size_bin_idx,'matched': False})
    det_boxes_list.sort(key=lambda x: x[5], reverse=True); det_matched_status=[{'matched': False,'matched_gt_idx':-1} for _ in range(len(det_boxes_list))]
    for det_idx, det in enumerate(det_boxes_list):
        det_box=det[:4]; det_cls=int(det[4]); det_area=get_box_area(det_box); is_potato_pred=det_cls == potato_cls_id; is_impurity_pred=det_cls in impurity_cls_ids; det_size_bin_idx=-1
        for idx, (lower, upper) in enumerate(size_bins):
             if lower <= det_area < upper: det_size_bin_idx = idx; break
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
            for idx, (lower, upper) in enumerate(size_bins):
                 if lower <= det_area < upper: det_size_bin_idx = idx; break
            stats['per_class'][det_cls]['fp']+=1
            if det_size_bin_idx!=-1: stats['per_size_bin'][det_size_bin_idx]['fp']+=1; stats['per_class_size_bin'][det_cls][det_size_bin_idx]['fp']+=1
            is_potato_pred = det_cls == potato_cls_id; is_impurity_pred = det_cls in impurity_cls_ids
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

# Calculate Derived Metrics (Same as YOLOv12 script)
def calculate_derived_metrics(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'tp': tp, 'fp': fp, 'fn': fn}

# --- Worker function for parallel processing ---
# (Keep evaluate_combination_yolox from previous version, uses new comprehensive eval)
def evaluate_combination_yolox(combo, all_detections_raw_dict_yolox, all_ground_truths,
                               class_names, potato_cls_name, impurity_cls_names,
                               num_classes, iou_threshold, size_bins):
    p_conf, i_conf, p_nms, i_nms = combo
    potato_conf=round(p_conf,3); impurity_conf=round(i_conf,3); potato_nms=round(p_nms,3); impurity_nms=round(i_nms,3)
    total_stats = {
        'overall_potato': {'tp':0,'fp':0,'fn':0},'overall_impurity': {'tp':0,'fp':0,'fn':0},
        'per_class': defaultdict(lambda: {'tp':0,'fp':0,'fn':0}),'per_size_bin': defaultdict(lambda: {'tp':0,'fp':0,'fn':0}),
        'per_class_size_bin': defaultdict(lambda: defaultdict(lambda: {'tp':0,'fp':0,'fn':0})) }
    class_id_map = {name: i for i, name in enumerate(class_names)} # Use map for consistency
    for img_id, raw_dets_tensor in all_detections_raw_dict_yolox.items():
        gt_boxes = all_ground_truths.get(img_id, np.array([]))
        filtered_dets = apply_confidence_and_nms_yolox(raw_dets_tensor, potato_conf, impurity_conf, potato_nms, impurity_nms, class_names, potato_cls_name, impurity_cls_names)
        img_stats = evaluate_precision_recall_comprehensive_yolox(filtered_dets, gt_boxes, num_classes, iou_threshold, class_names, potato_cls_name, impurity_cls_names, size_bins)
        for category, metrics in img_stats.items():
             if category in ['overall_potato','overall_impurity']: total_stats[category]['tp']+=metrics['tp']; total_stats[category]['fp']+=metrics['fp']; total_stats[category]['fn']+=metrics['fn']
             elif category=='per_class':
                  for cls_id, class_metrics in metrics.items(): total_stats[category][cls_id]['tp']+=class_metrics['tp']; total_stats[category][cls_id]['fp']+=class_metrics['fp']; total_stats[category][cls_id]['fn']+=class_metrics['fn']
             elif category=='per_size_bin':
                  for bin_idx, bin_metrics in metrics.items(): total_stats[category][bin_idx]['tp']+=bin_metrics['tp']; total_stats[category][bin_idx]['fp']+=bin_metrics['fp']; total_stats[category][bin_idx]['fn']+=bin_metrics['fn']
             elif category=='per_class_size_bin':
                  for cls_id, class_bins in metrics.items():
                      for bin_idx, bin_metrics in class_bins.items(): total_stats[category][cls_id][bin_idx]['tp']+=bin_metrics['tp']; total_stats[category][cls_id][bin_idx]['fp']+=bin_metrics['fp']; total_stats[category][cls_id][bin_idx]['fn']+=bin_metrics['fn']
    results = {'potato_conf': potato_conf,'impurity_conf': impurity_conf,'potato_nms': potato_nms,'impurity_nms': impurity_nms}
    overall_potato=calculate_derived_metrics(**total_stats['overall_potato']); overall_impurity=calculate_derived_metrics(**total_stats['overall_impurity'])
    results.update({f'overall_potato_{k}':v for k,v in overall_potato.items()}); results.update({f'overall_impurity_{k}':v for k,v in overall_impurity.items()})
    for cls_id, counts in total_stats['per_class'].items():
        if 0 <= cls_id < len(class_names):
             class_name=class_names[cls_id]; safe_class_name="".join(c if c.isalnum() else "_" for c in class_name); metrics=calculate_derived_metrics(**counts); results.update({f'class_{safe_class_name}_{k}':v for k,v in metrics.items()})
    for bin_idx, counts in total_stats['per_size_bin'].items():
        size_label=get_size_bin_label(0,[size_bins[bin_idx]]); safe_size_label=size_label # Already safe format
        metrics=calculate_derived_metrics(**counts); results.update({f'size_{safe_size_label}_{k}':v for k,v in metrics.items()})
    for cls_id, class_bins in total_stats['per_class_size_bin'].items():
        if 0 <= cls_id < len(class_names):
             class_name=class_names[cls_id]; safe_class_name="".join(c if c.isalnum() else "_" for c in class_name)
             for bin_idx, counts in class_bins.items():
                  size_label=get_size_bin_label(0,[size_bins[bin_idx]]); safe_size_label=size_label
                  metrics=calculate_derived_metrics(**counts); results.update({f'class_{safe_class_name}_size_{safe_size_label}_{k}':v for k,v in metrics.items()})
    small_bin_overall_metrics=calculate_derived_metrics(**total_stats['per_size_bin'].get(0,{'tp':0,'fp':0,'fn':0})); results['small_overall_f1']=small_bin_overall_metrics['f1']
    # Add PMR/IDR in % format for potential direct use/comparison with original script's reporting style
    potato_precision = results.get('overall_potato_precision', 0.0); impurity_recall = results.get('overall_impurity_recall', 0.0)
    results['potato_misclass_rate'] = (1.0 - potato_precision) * 100.0; results['impurity_recall_percent'] = impurity_recall * 100.0
    return results

# --- Visualization Functions ---
# (Reuse visualize_confusion_matrix, visualize_comparison, get_detection_signature,
#  generate_analysis_plots, plot_size_bin_performance from YOLOv12 script)

def visualize_confusion_matrix(tp_potato, fp_potato, fn_potato,
                              tp_impurity, fp_impurity, fn_impurity,
                              output_dir, title_suffix=""):
    # (Identical to the one in yolov12_evaluation_v5_1_trt.py)
    output_vis_dir = os.path.join(output_dir, "confusion_matrix")
    os.makedirs(output_vis_dir, exist_ok=True)
    try:
        cm = np.array([[int(tp_potato), int(fp_impurity)],[int(fp_potato), int(tp_impurity)]], dtype=int)
    except ValueError as e: logger.error(f"CM Value Error: {e}"); return
    class_names_plot = ["Potato", "Impurity"]
    try:
        plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names_plot,yticklabels=class_names_plot)
        plt.xlabel('Predicted Class'); plt.ylabel('True Class'); plt.title(f'Confusion Matrix - Counts{title_suffix}'); plt.tight_layout(); cm_path=os.path.join(output_vis_dir,f"confusion_matrix_absolute{title_suffix.replace(' ','_')}.png"); plt.savefig(cm_path); plt.close(); logger.info(f"Saved counts CM to {cm_path}")
        cm_percent=np.zeros_like(cm,dtype=float); row_sums=cm.sum(axis=1)[:,np.newaxis]; non_zero_rows=row_sums>0;
        if non_zero_rows.any(): cm_percent[non_zero_rows[:,0]]=cm[non_zero_rows[:,0]]/row_sums[non_zero_rows]*100
        plt.figure(figsize=(8,6)); sns.heatmap(cm_percent,annot=True,fmt=".2f",cmap="Blues",xticklabels=class_names_plot,yticklabels=class_names_plot,vmin=0,vmax=100)
        plt.xlabel('Predicted Class'); plt.ylabel('True Class'); plt.title(f'Confusion Matrix - % (Recall Focus){title_suffix}'); plt.tight_layout(); cm_perc_path=os.path.join(output_vis_dir,f"confusion_matrix_percent{title_suffix.replace(' ','_')}.png"); plt.savefig(cm_perc_path); plt.close(); logger.info(f"Saved percentage CM to {cm_perc_path}")
    except Exception as e: logger.error(f"Failed CM plot generation: {e}")

def visualize_comparison(image_id, original_img_path, standard_dets, optimal_dets, gt_boxes,
                         standard_thresholds, optimal_thresholds, class_names_map, output_vis_dir):
    # (Identical to the one in yolov12_evaluation_v5_1_trt.py, including TP/FP/FN labeling)
    original_img = cv2.imread(str(original_img_path));
    if original_img is None: logger.warning(f"Could not read image: {original_img_path}"); return
    vis_img = original_img.copy(); h,w = vis_img.shape[:2]
    color_std_only_tp=(0,0,200); color_std_only_fp=(150,0,0); color_opt_only_tp=(0,200,0); color_opt_only_fp=(0,150,0)
    color_common_tp=(200,150,150); color_common_fp=(150,150,200); color_gt=(0,0,255); color_fn=(0,165,255)
    thickness_unique=2; thickness_common=1; thickness_gt=1; thickness_fn=1; font_scale=0.4; font=cv2.FONT_HERSHEY_SIMPLEX; text_color_dark_bg=(255,255,255); text_color_light_bg=(0,0,0)
    gt_info_vis = [{'box': gt[:4],'cls_id': int(gt[4]),'matched_std': False,'matched_opt': False} for gt in gt_boxes if len(gt)>=5]
    det_info_std = [{'det': det,'is_tp': False} for det in standard_dets]; det_info_opt = [{'det': det,'is_tp': False} for det in optimal_dets]
    temp_gt_matched_flags=[False]*len(gt_info_vis); det_info_opt.sort(key=lambda x:x['det'][5],reverse=True)
    for det_item in det_info_opt:
        best_iou=0.0; best_gt_idx=-1; det_box=det_item['det'][:4]; det_cls=int(det_item['det'][4])
        for gt_idx, gt_item in enumerate(gt_info_vis):
            if not temp_gt_matched_flags[gt_idx] and gt_item['cls_id']==det_cls:
                iou=calculate_iou(det_box,gt_item['box']);
                if iou>=IOU_THRESHOLD and iou>best_iou: best_iou=iou; best_gt_idx=gt_idx
        if best_gt_idx!=-1: det_item['is_tp']=True; gt_info_vis[best_gt_idx]['matched_opt']=True; temp_gt_matched_flags[best_gt_idx]=True
    temp_gt_matched_flags=[False]*len(gt_info_vis); det_info_std.sort(key=lambda x:x['det'][5],reverse=True)
    for det_item in det_info_std:
        best_iou=0.0; best_gt_idx=-1; det_box=det_item['det'][:4]; det_cls=int(det_item['det'][4])
        for gt_idx, gt_item in enumerate(gt_info_vis):
             if not temp_gt_matched_flags[gt_idx] and gt_item['cls_id']==det_cls:
                 iou=calculate_iou(det_box,gt_item['box']);
                 if iou>=IOU_THRESHOLD and iou>best_iou: best_iou=iou; best_gt_idx=gt_idx
        if best_gt_idx!=-1: det_item['is_tp']=True; gt_info_vis[best_gt_idx]['matched_std']=True; temp_gt_matched_flags[best_gt_idx]=True
    standard_sig=get_detection_signature(standard_dets); optimal_sig=get_detection_signature(optimal_dets); drawn_common_signatures=set()
    for det_item in det_info_std:
        det=det_item['det']; is_tp=det_item['is_tp']; x1,y1,x2,y2,cls_id,score=det; cls_id=int(cls_id); pt1,pt2=(int(x1),int(y1)),(int(x2),int(y2)); det_sig=(cls_id,round(x1),round(y1),round(x2),round(y2)); base_label=f"S:{class_names_map.get(cls_id,'UNK')} {score:.2f}"; label=base_label+(" [TP]" if is_tp else " [FP]")
        (lbl_w,lbl_h),_=cv2.getTextSize(label,font,font_scale,1); text_pt=(pt1[0],pt1[1]-2); text_bg_pt2=(pt1[0]+lbl_w,pt1[1]-lbl_h-3)
        if det_sig not in optimal_sig: color=color_std_only_tp if is_tp else color_std_only_fp; cv2.rectangle(vis_img,pt1,pt2,color,thickness_unique); cv2.rectangle(vis_img,(pt1[0],pt1[1]),text_bg_pt2,color,-1); cv2.putText(vis_img,label,text_pt,font,font_scale,text_color_dark_bg,1,cv2.LINE_AA)
        elif det_sig not in drawn_common_signatures: color=color_common_tp if is_tp else color_common_fp; cv2.rectangle(vis_img,pt1,pt2,color,thickness_common); cv2.rectangle(vis_img,(pt1[0],pt1[1]),text_bg_pt2,color,-1); cv2.putText(vis_img,label,text_pt,font,font_scale,text_color_light_bg,1,cv2.LINE_AA); drawn_common_signatures.add(det_sig)
    for det_item in det_info_opt:
        det=det_item['det']; is_tp=det_item['is_tp']; x1,y1,x2,y2,cls_id,score=det; cls_id=int(cls_id); pt1,pt2=(int(x1),int(y1)),(int(x2),int(y2)); det_sig=(cls_id,round(x1),round(y1),round(x2),round(y2))
        if det_sig not in standard_sig:
            base_label=f"O:{class_names_map.get(cls_id,'UNK')} {score:.2f}"; label=base_label+(" [TP]" if is_tp else " [FP]"); color=color_opt_only_tp if is_tp else color_opt_only_fp
            (lbl_w,lbl_h),_=cv2.getTextSize(label,font,font_scale,1); text_y=pt1[1]-lbl_h-3 if pt1[1]>(lbl_h+10) else pt2[1]+lbl_h+13; text_bg_pt1=(pt1[0],text_y-lbl_h-1); text_bg_pt2=(pt1[0]+lbl_w,text_y+1); text_pt=(pt1[0],text_y)
            cv2.rectangle(vis_img,pt1,pt2,color,thickness_unique); cv2.rectangle(vis_img,text_bg_pt1,text_bg_pt2,color,-1); cv2.putText(vis_img,label,text_pt,font,font_scale,text_color_light_bg,1,cv2.LINE_AA)
    for gt_item in gt_info_vis:
        gt_box=gt_item['box']; cls_id=gt_item['cls_id']; pt1,pt2=(int(gt_box[0]),int(gt_box[1])),(int(gt_box[2]),int(gt_box[3]))
        cv2.rectangle(vis_img,pt1,pt2,color_gt,thickness_gt)
        if not gt_item['matched_opt']: label="FN"; (lbl_w,lbl_h),_=cv2.getTextSize(label,font,font_scale,1); text_pt=(pt2[0]-lbl_w-2,pt2[1]-2); cv2.putText(vis_img,label,text_pt,font,font_scale+0.1,color_fn,1,cv2.LINE_AA)
    legend_y=20; line_height=16; font_size=0.4; std_pc,std_ic,std_pn,std_in=standard_thresholds; opt_pc,opt_ic,opt_pn,opt_in=optimal_thresholds
    cv2.putText(vis_img,"Legend:",(5,legend_y),font,font_size,(0,0,0),1,cv2.LINE_AA); legend_y+=line_height; cv2.putText(vis_img,"- TP (Std Only): Dark Blue",(10,legend_y),font,font_size,color_std_only_tp,1,cv2.LINE_AA); legend_y+=line_height; cv2.putText(vis_img,"- FP (Std Only): Light Blue",(10,legend_y),font,font_size,color_std_only_fp,1,cv2.LINE_AA); legend_y+=line_height
    cv2.putText(vis_img,"- TP (Opt Only): Dark Green",(10,legend_y),font,font_size,color_opt_only_tp,1,cv2.LINE_AA); legend_y+=line_height; cv2.putText(vis_img,"- FP (Opt Only): Light Green",(10,legend_y),font,font_size,color_opt_only_fp,1,cv2.LINE_AA); legend_y+=line_height
    cv2.putText(vis_img,"- TP (Common): Cyan/Pink",(10,legend_y),font,font_size,color_common_tp,1,cv2.LINE_AA); legend_y+=line_height; cv2.putText(vis_img,"- FP (Common): Purple",(10,legend_y),font,font_size,color_common_fp,1,cv2.LINE_AA); legend_y+=line_height
    cv2.putText(vis_img,"- GT Box: Red",(10,legend_y),font,font_size,color_gt,1,cv2.LINE_AA); legend_y+=line_height; cv2.putText(vis_img,"- FN (Opt Missed): Orange Text",(10,legend_y),font,font_size,color_fn,1,cv2.LINE_AA); legend_y+=line_height; legend_y+=5
    cv2.putText(vis_img,f"Std: Pc={std_pc:.2f},Ic={std_ic:.2f},Pn={std_pn:.2f},In={std_in:.2f}",(5,legend_y),font,font_size,(0,0,0),1,cv2.LINE_AA); legend_y+=line_height; cv2.putText(vis_img,f"Opt: Pc={opt_pc:.3f},Ic={opt_ic:.3f},Pn={opt_pn:.3f},In={opt_in:.3f}",(5,legend_y),font,font_size,(0,0,0),1,cv2.LINE_AA);
    save_path=os.path.join(output_vis_dir,f"{image_id}_comparison_enhanced.jpg");
    try: cv2.imwrite(save_path, vis_img)
    except Exception as e: logger.error(f"Failed save enhanced comparison: {save_path}: {e}")

def get_detection_signature(detections):
    signature=set();
    for det in detections: x1,y1,x2,y2,cls_id,score=det; sig_tuple=(int(cls_id),round(x1),round(y1),round(x2),round(y2)); signature.add(sig_tuple)
    return signature

def generate_analysis_plots(results_df, output_dir, class_names_map, size_bins):
    # (Identical to the one in yolov12_evaluation_v5_1_trt.py)
    plot_dir = os.path.join(output_dir, ANALYSIS_PLOTS_DIR); os.makedirs(plot_dir, exist_ok=True); logger.info(f"Generating analysis plots in: {plot_dir}")
    if results_df.empty: logger.warning("Results DataFrame empty. Skipping analysis plots."); return
    metrics_to_heatmap = {'overall_potato_precision': 'Overall Potato Precision (PMR Goal)', 'overall_impurity_recall': 'Overall Impurity Recall (IDR Goal)', 'small_overall_f1': 'Small Object F1 (Overall, <3k)', }
    try:
        results_df['potato_conf_r']=results_df['potato_conf'].round(3); results_df['impurity_conf_r']=results_df['impurity_conf'].round(3); results_df['potato_nms_r']=results_df['potato_nms'].round(3); results_df['impurity_nms_r']=results_df['impurity_nms'].round(3)
        grouped = results_df.groupby(['potato_conf_r','impurity_conf_r'])
        for metric_col, title in metrics_to_heatmap.items():
             if metric_col not in results_df.columns: logger.warning(f"Metric '{metric_col}' not found for heatmap."); continue
             try:
                 avg_metric_df = grouped[metric_col].mean().reset_index(); pivot_df = avg_metric_df.pivot_table(index='impurity_conf_r', columns='potato_conf_r', values=metric_col)
                 plt.figure(figsize=(10,8)); sns.heatmap(pivot_df.sort_index(ascending=False),annot=True,fmt=".3f",cmap="viridis",linewidths=.5,cbar_kws={'label':f"Avg. {title}"})
                 plt.xlabel("Potato Confidence Threshold"); plt.ylabel("Impurity Confidence Threshold"); plt.title(f"Heatmap vs Conf (Avg over NMS): {title}"); plt.tight_layout(); plot_path=os.path.join(plot_dir,f"heatmap_avgNMS_{metric_col}.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved heatmap (Avg NMS): {plot_path}")
             except Exception as e: logger.error(f"Failed heatmap for {metric_col}: {e}", exc_info=False)
    except Exception as e: logger.error(f"Error heatmap prep: {e}", exc_info=True)
    try:
        plt.figure(figsize=(10,8)); scatter=plt.scatter(results_df['overall_impurity_recall'],results_df['overall_potato_precision'],c=results_df['small_overall_f1'],cmap='viridis',alpha=0.6,s=30)
        plt.colorbar(scatter,label='Small Object F1 (Overall, <3k)'); plt.axhline(TARGET_POTATO_PRECISION,color='red',linestyle='--',linewidth=1,label=f'Target Potato P = {TARGET_POTATO_PRECISION}')
        plt.axvline(TARGET_IMPURITY_RECALL,color='blue',linestyle='--',linewidth=1,label=f'Target Impurity R = {TARGET_IMPURITY_RECALL}')
        plt.fill_betweenx(y=[TARGET_POTATO_PRECISION, 1.01], x1=TARGET_IMPURITY_RECALL, x2=1.01, color='green', alpha=0.1, label='Target Zone')
        plt.xlabel("Overall Impurity Recall (IDR)"); plt.ylabel("Overall Potato Precision (1-PMR)"); plt.title("Performance Trade-off (Overall Metrics)")
        min_r=max(0,results_df['overall_impurity_recall'].min()-0.05); min_p=max(0.9,results_df['overall_potato_precision'].min()-0.01)
        plt.xlim(left=min_r, right=1.01); plt.ylim(bottom=min_p, top=1.005); plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='lower right'); plt.tight_layout(); plot_path=os.path.join(plot_dir,"pr_scatter_overall_goals.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved P-R scatter plot (Overall Goals): {plot_path}")
    except Exception as e: logger.error(f"Failed P-R scatter plot: {e}", exc_info=True)

def plot_size_bin_performance(best_result_dict, class_names_map, size_bins, output_dir):
    # (Identical to the one in yolov12_evaluation_v5_1_trt.py, uses class_names_map)
    plot_dir = os.path.join(output_dir, ANALYSIS_PLOTS_DIR); logger.info("Generating size bin performance plot...");
    if not best_result_dict: logger.warning("No best result data to plot size bin performance."); return
    data = []; class_id_to_name = class_names_map # Use map directly
    for cls_id, cls_name in class_id_to_name.items():
        safe_cls_name = "".join(c if c.isalnum() else "_" for c in cls_name)
        for bin_idx, size_bin in enumerate(size_bins):
             size_label_safe = get_size_bin_label(0, [size_bin]); size_label_plot = size_label_safe.replace('_','-').replace('lt','<').replace('gte','>=') # For plotting
             prefix = f'class_{safe_cls_name}_size_{size_label_safe}_' # Use safe label for dict key lookup
             if f'{prefix}f1' in best_result_dict:
                  data.append({'Class':cls_name,'Size Bin':size_label_plot,'Precision':best_result_dict.get(f'{prefix}precision',0.0),'Recall':best_result_dict.get(f'{prefix}recall',0.0),'F1':best_result_dict.get(f'{prefix}f1',0.0),'TP':best_result_dict.get(f'{prefix}tp',0),'Bin Index':bin_idx})
    if not data: logger.warning("No valid size bin data found in best result."); return
    df = pd.DataFrame(data); df = df.sort_values(by=['Class', 'Bin Index'])
    try:
        plt.figure(figsize=(14,8)); sns.barplot(data=df,x='Size Bin',y='F1',hue='Class',palette='viridis')
        plt.xlabel("Object Size Bin (pixels²)"); plt.ylabel("F1-Score"); plt.title(f"F1-Score by Object Size and Class (YOLOX Optimal Thresholds)"); plt.xticks(rotation=45, ha='right'); plt.ylim(0,1.05); plt.grid(axis='y',linestyle=':',alpha=0.7); plt.legend(title='Class',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout()
        plot_path = os.path.join(plot_dir, "bar_size_bin_f1_yolox_optimal.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved YOLOX size bin F1 plot: {plot_path}")
    except Exception as e: logger.error(f"Failed YOLOX size bin plot: {e}", exc_info=True)

# --- Logging Setup ---
def setup_logging(output_dir):
    logger.remove(); log_format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>|<level>{level:<8}</level>|<level>{message}</level>"; logger.add(sys.stderr,level="INFO",format=log_format)
    log_file_path=os.path.join(output_dir,"evaluation_log_yolox_v2_trt.log"); logger.add(log_file_path,level="DEBUG",format=log_format,rotation="10 MB")

# --- JSON Serialization Helper ---
# (Keep robust make_serializable from YOLOv12 script)
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
def calculate_map_yolox(exp_file, model_path, device, tsize):
    """ Placeholder for YOLOX mAP calculation using its standard tools. """
    logger.info("Calculating mAP@0.5 for YOLOX (Requires separate YOLOX eval tools)...")
    # Requires running tools/eval.py, e.g.:
    # python tools/eval.py -f {exp_file} -c {model_path} -b 8 -d 1 --conf 0.001 --nms 0.65 --tsize {tsize} [--trt if model_path is TRT]
    logger.warning("mAP calculation needs separate run of YOLOX tools/eval.py.")
    return -1.0 # Placeholder

# --- Main Execution Logic ---
def make_parser():
    # (Keep make_parser from original, ensure --trt_path is useful)
    parser = argparse.ArgumentParser("YOLOX Comprehensive TensorRT Evaluation (v2-trt)")
    parser.add_argument("-f", "--exp_file", default="exps/default/yolox_s.py", type=str, help="Path to YOLOX experiment file (needed for config).")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="Path to PyTorch checkpoint file (.pth) (Use --trt or --trt_path instead).")
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="Model name (used for finding TRT file path if not explicit).")
    parser.add_argument("--trt", action="store_true", help="Use TensorRT model (Recommended). Expects YOLOX_outputs/NAME/model_trt.pth or --trt_path.")
    parser.add_argument("--trt_path", type=str, default=None, help="Explicit path to YOLOX TensorRT engine file (overrides default).")
    parser.add_argument("--tsize", default=640, type=int, help="Test image size (square).")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision (primarily for TRT).")
    parser.add_argument("--legacy", action="store_true", help="Use legacy YOLOX pre-processing.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing test images.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory containing VOC XML annotations.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save results.")
    parser.add_argument("--raw_data_file", type=str, default=RAW_DATA_FILENAME, help="Filename for raw detection data within output_dir.")
    parser.add_argument("--analyze_only", action="store_true", help="Skip inference, only run analysis on existing raw_data_file.")
    parser.add_argument("--save_vis_raw", action="store_true", help="Save visualization images during raw data collection.")
    parser.add_argument("--save_vis_comp", action="store_true", help="Save enhanced comparison visualization images after analysis.")
    parser.add_argument("--device", default="gpu", type=str, help="Device to run on ('gpu' or 'cpu').")
    return parser

def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logger.info("Starting YOLOX Comprehensive Evaluation v2 (TensorRT Focused)")
    logger.info(f"Output directory: {args.output_dir}")

    # --- TRT Focus Check ---
    if not args.trt and not args.trt_path:
        logger.error("This script requires a TensorRT model. Please use --trt or provide --trt_path.")
        sys.exit(1)
    if args.ckpt and (args.trt or args.trt_path):
        logger.warning("Both --ckpt and --trt/--trt_path specified. Prioritizing TRT.")
        args.ckpt = None # Ignore checkpoint if TRT is specified

    # Ensure class names are consistent
    class_names = CLASS_NAMES
    class_names_map = {i: name for i, name in enumerate(class_names)} # For visualization
    num_classes = len(class_names)
    logger.info(f"Using Class Names: {class_names}")

    raw_data_filepath=os.path.join(args.output_dir,args.raw_data_file); results_filepath=os.path.join(args.output_dir,RESULTS_FILENAME); best_result_filepath=os.path.join(args.output_dir,BEST_RESULT_FILENAME); comparison_vis_dir=os.path.join(args.output_dir,COMPARISON_VIS_DIR); analysis_plots_dir=os.path.join(args.output_dir,ANALYSIS_PLOTS_DIR)

    try: exp=get_exp(args.exp_file,args.name)
    except Exception as e: logger.error(f"Failed load exp file {args.exp_file}: {e}"); sys.exit(1)
    if args.tsize is not None: exp.test_size=(args.tsize,args.tsize)
    exp.nmsthre=0.65 # NMS threshold during raw data collection postprocess

    all_detections_raw_dict=None; all_ground_truths=None

    if args.analyze_only:
        logger.info("Analyze only mode: Loading existing YOLOX raw data...")
        raw_data=load_raw_data(raw_data_filepath)
        if raw_data:
            all_detections_raw_dict=raw_data.get("detections"); all_ground_truths=raw_data.get("ground_truths"); loaded_class_names=raw_data.get("class_names",class_names)
            if loaded_class_names!=class_names: logger.warning(f"Class name mismatch! Using current: {class_names}")
            else: class_names=loaded_class_names # Use names from saved data if consistent
            class_names_map = {i: name for i, name in enumerate(class_names)} # Update map if loaded
            num_classes = len(class_names)
        if not all_detections_raw_dict or not all_ground_truths: logger.error("Failed load/incomplete YOLOX raw data. Exiting."); sys.exit(1)
    else:
        logger.info("Starting YOLOX raw data collection phase...")
        # --- Model Loading (TRT Path) ---
        trt_file_path=args.trt_path or os.path.join("YOLOX_outputs",args.name,"model_trt.pth")
        logger.info(f"Using TRT model path: {trt_file_path}")
        if not os.path.exists(trt_file_path): logger.error(f"TRT model not found: {trt_file_path}"); sys.exit(1)
        try: model_structure = exp.get_model() # Need structure for predictor
        except Exception as e: logger.error(f"Failed get model structure: {e}"); sys.exit(1)
        decoder=None # Configure decoder if needed for YOLOX TRT

        # --- Initialize Predictor ---
        target_device="gpu" if args.device=="gpu" else "cpu"
        try:
            predictor=TRTPredictor(model=model_structure,exp=exp,cls_names=class_names,trt_file=trt_file_path,decoder=decoder,device=target_device,fp16=args.fp16,legacy=args.legacy)
            logger.info("YOLOX Predictor initialized successfully (using TRT).")
        except Exception as e: logger.error(f"Failed init YOLOX predictor: {e}"); sys.exit(1)

        # --- Run Inference Loop ---
        annotation_loader=VOCAnnotationLoader(args.annotations_dir,class_names)
        try: image_files=[f for f in os.listdir(args.images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))];
        except FileNotFoundError: logger.error(f"Images dir not found: {args.images_dir}"); sys.exit(1)
        if not image_files: logger.error(f"No images in {args.images_dir}"); sys.exit(1)
        logger.info(f"Found {len(image_files)} images.")

        all_detections_raw_dict={}; all_ground_truths={}; inference_times=[]
        logger.info("Running YOLOX inference for raw detections...")
        for filename in tqdm(image_files,desc="Collecting Raw Data (YOLOX TRT)"):
             image_path=os.path.join(args.images_dir,filename); image_id=os.path.splitext(filename)[0]
             gt_boxes=annotation_loader.get_annotation(image_id)
             all_ground_truths[image_id]=np.array(gt_boxes,dtype=np.float32) if len(gt_boxes)>0 else np.empty((0,5),dtype=np.float32)
             img=cv2.imread(image_path)
             if img is None: logger.warning(f"Skip unloadable: {image_path}"); all_detections_raw_dict[image_id]=None; continue
             raw_dets_tensor,img_info=predictor.inference(img)
             all_detections_raw_dict[image_id]=raw_dets_tensor # Store raw tensor
             if raw_dets_tensor is not None: inference_times.append(img_info["inference_time"])
             else: logger.warning(f"No dets/error for: {filename}")
             if args.save_vis_raw and raw_dets_tensor is not None:
                  try:
                      vis_img=img.copy(); temp_processed_dets=apply_confidence_and_nms_yolox(raw_dets_tensor,0.01,0.01,0.9,0.9,class_names,POTATO_CLS_NAME,IMPURITY_CLS_NAMES)
                      for x1,y1,x2,y2,cls_id,score in temp_processed_dets: cv2.rectangle(vis_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                      for g_x1,g_y1,g_x2,g_y2,g_cls_id in all_ground_truths[image_id]: cv2.rectangle(vis_img,(int(g_x1),int(g_y1)),(int(g_x2),int(g_y2)),(0,255,0),1)
                      vis_path=os.path.join(args.output_dir,f"{image_id}_raw_vis_yolox.jpg"); cv2.imwrite(vis_path,vis_img)
                  except Exception as vis_e: logger.warning(f"Failed raw vis YOLOX {image_id}: {vis_e}")
        data_to_save={"detections":all_detections_raw_dict,"ground_truths":all_ground_truths,"class_names":class_names}
        save_raw_data(data_to_save,raw_data_filepath)
        avg_inf_time=sum(inference_times)/len(inference_times) if inference_times else 0; fps=1.0/avg_inf_time if avg_inf_time>0 else 0
        logger.info(f"YOLOX Raw data collection done. Avg inference: {avg_inf_time:.4f}s ({fps:.2f} FPS)")

    # --- Offline Analysis Phase ---
    if not all_detections_raw_dict or not all_ground_truths: logger.error("Missing YOLOX raw data for analysis."); sys.exit(1)

    logger.info("Starting comprehensive offline analysis with NMS tuning (YOLOX)...")
    combinations=[(round(pc,3),round(ic,3),round(pn,3),round(imn,3)) for pc in POTATO_CONF_RANGE for ic in IMPURITY_CONF_RANGE for pn in POTATO_NMS_RANGE for imn in IMPURITY_NMS_RANGE]
    logger.info(f"Total combinations: {len(combinations)}")
    if not combinations: logger.error("No combinations generated."); sys.exit(1)
    all_results=[]
    try:
        num_workers=os.cpu_count(); logger.info(f"Starting parallel evaluation ({num_workers} workers)...")
        worker_func=functools.partial(evaluate_combination_yolox,all_detections_raw_dict_yolox=all_detections_raw_dict,all_ground_truths=all_ground_truths,class_names=class_names,potato_cls_name=POTATO_CLS_NAME,impurity_cls_names=IMPURITY_CLS_NAMES,num_classes=num_classes,iou_threshold=IOU_THRESHOLD,size_bins=SIZE_BINS)
        with multiprocessing.Pool(processes=num_workers) as pool:
             results_iterator=pool.imap_unordered(worker_func,combinations)
             for result in tqdm(results_iterator,total=len(combinations),desc="Analyzing Thresholds (YOLOX)"):
                 if result is not None: all_results.append(result)
        logger.info(f"Finished parallel evaluation. Collected {len(all_results)} YOLOX results.")
    except KeyboardInterrupt: logger.warning("Analysis interrupted."); sys.exit(1)
    except Exception as pool_e: logger.error(f"Multiprocessing failed: {pool_e}"); sys.exit(1)
    if not all_results: logger.error("No YOLOX results collected."); sys.exit(1)

    # --- Save All Results & Generate Analysis Plots ---
    logger.info(f"Saving all {len(all_results)} YOLOX results to {results_filepath}...")
    try:
        results_df=pd.DataFrame(all_results); results_df.sort_values(by=['potato_conf','impurity_conf','potato_nms','impurity_nms'],inplace=True)
        for col in results_df.select_dtypes(include=np.number).columns: results_df[col]=pd.to_numeric(results_df[col],errors='coerce')
        results_df.to_csv(results_filepath,index=False,float_format='%.6f'); logger.info("Saved all YOLOX results to CSV.")
        generate_analysis_plots(results_df,args.output_dir,class_names_map,SIZE_BINS) # Use adapted plotting
    except Exception as e: logger.error(f"Failed save/plot YOLOX results: {e}");
    if 'results_df' not in locals(): results_df=pd.DataFrame(all_results) # Keep data in memory

    # --- Selection Logic (Using OVERALL metrics) ---
    logger.info("Selecting best YOLOX threshold based on OVERALL goals...")
    if results_df.empty: logger.error("Results DF empty for selection."); sys.exit(1)
    candidates_df=results_df[(results_df['overall_potato_precision']>=TARGET_POTATO_PRECISION)&(results_df['overall_impurity_recall']>=TARGET_IMPURITY_RECALL)].copy()
    logger.info(f"Found {len(candidates_df)} YOLOX combinations meeting OVERALL goals.")
    best_result_dict=None
    if not candidates_df.empty:
        candidates_df=candidates_df.sort_values(by=['small_overall_f1','overall_impurity_recall'],ascending=[False,False])
        best_result_dict=candidates_df.iloc[0].to_dict(); logger.info(f"Selected best YOLOX: Conf_P={best_result_dict['potato_conf']:.3f}, Conf_I={best_result_dict['impurity_conf']:.3f}, NMS_P={best_result_dict['potato_nms']:.3f}, NMS_I={best_result_dict['impurity_nms']:.3f}")
    else: logger.warning("No YOLOX combination met OVERALL goals!")

    # --- Reporting and Final Visualization ---
    if best_result_dict:
        logger.info("--- Final Optimal YOLOX Result (Overall Goals & Small F1) ---")
        pc=best_result_dict['potato_conf'];ic=best_result_dict['impurity_conf'];pn=best_result_dict['potato_nms'];imn=best_result_dict['impurity_nms']
        logger.info(f"Thresholds: Pc={pc:.3f}, Ic={ic:.3f}, Pn={pn:.3f}, In={imn:.3f}")
        logger.info(" Metrics (Overall):")
        logger.info(f"  Potato Precision: {best_result_dict.get('overall_potato_precision',-1):.6f} (Target >= {TARGET_POTATO_PRECISION})")
        logger.info(f"  Impurity Recall:  {best_result_dict.get('overall_impurity_recall',-1):.6f} (Target >= {TARGET_IMPURITY_RECALL})")
        logger.info(f"  Small Obj F1 (<3k):{best_result_dict.get('small_overall_f1',-1):.6f}")
        logger.info(" Metrics (Per Class - Overall Size):")
        for cls_id, cls_name in enumerate(class_names):
             safe_cls_name="".join(c if c.isalnum() else"_" for c in cls_name); p_key,r_key,f1_key=f"class_{safe_cls_name}_precision",f"class_{safe_cls_name}_recall",f"class_{safe_cls_name}_f1"
             if p_key in best_result_dict: logger.info(f"  {cls_name}: P={best_result_dict[p_key]:.4f}, R={best_result_dict[r_key]:.4f}, F1={best_result_dict[f1_key]:.4f}")
        try: # Save best result JSON
             logger.info(f"Saving best YOLOX result to {best_result_filepath}..."); serializable_best=make_serializable(best_result_dict);
             with open(best_result_filepath,'w') as f: json.dump(serializable_best,f,indent=4); logger.info("Best YOLOX result saved.")
        except Exception as e: logger.error(f"Failed save best YOLOX result: {e}")
        # Visualize Confusion Matrix
        visualize_confusion_matrix(tp_potato=best_result_dict.get('overall_potato_tp',0),fp_potato=best_result_dict.get('overall_impurity_fp',0),fn_potato=best_result_dict.get('overall_potato_fn',0),tp_impurity=best_result_dict.get('overall_impurity_tp',0),fp_impurity=best_result_dict.get('overall_potato_fp',0),fn_impurity=best_result_dict.get('overall_impurity_fn',0),output_dir=args.output_dir,title_suffix=" (YOLOX Optimal, Overall)")
        # Visualize Size Bin Performance
        plot_size_bin_performance(best_result_dict,class_names_map,SIZE_BINS,args.output_dir)
        # Enhanced Comparison Visualization (if enabled)
        if args.save_vis_comp:
             logger.info("Generating enhanced comparison visualizations for YOLOX...")
             os.makedirs(comparison_vis_dir,exist_ok=True); optimal_thresholds_tuple=(pc,ic,pn,imn); standard_thresholds_tuple=(STANDARD_POTATO_CONF,STANDARD_IMPURITY_CONF,STANDARD_POTATO_NMS,STANDARD_IMPURITY_NMS)
             vis_count=0; processed_count=0
             try: image_files_for_vis={p.stem: Path(args.images_dir)/p.name for p in Path(args.images_dir).glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']}
             except Exception as e: logger.error(f"Could not list images: {e}"); image_files_for_vis={}
             if image_files_for_vis:
                  for image_id in tqdm(all_detections_raw_dict.keys(),desc="Generating Enhanced Comparison Images (YOLOX)"):
                      processed_count+=1;
                      if image_id not in image_files_for_vis: continue
                      try:
                          raw_dets_tensor=all_detections_raw_dict[image_id]; gt_boxes=all_ground_truths.get(image_id,np.array([]))
                          # Get detections for standard and optimal thresholds
                          standard_dets=apply_confidence_and_nms_yolox(raw_dets_tensor,standard_thresholds_tuple[0],standard_thresholds_tuple[1],standard_thresholds_tuple[2],standard_thresholds_tuple[3],class_names,POTATO_CLS_NAME,IMPURITY_CLS_NAMES)
                          optimal_dets=apply_confidence_and_nms_yolox(raw_dets_tensor,optimal_thresholds_tuple[0],optimal_thresholds_tuple[1],optimal_thresholds_tuple[2],optimal_thresholds_tuple[3],class_names,POTATO_CLS_NAME,IMPURITY_CLS_NAMES)
                          standard_sig=get_detection_signature(standard_dets); optimal_sig=get_detection_signature(optimal_dets)
                          if standard_sig!=optimal_sig: vis_count+=1; visualize_comparison(image_id,image_files_for_vis[image_id],standard_dets,optimal_dets,gt_boxes,standard_thresholds_tuple,optimal_thresholds_tuple,class_names_map,comparison_vis_dir)
                      except Exception as e: logger.error(f"Error enhanced comparison YOLOX {image_id}: {e}",exc_info=False)
                  logger.info(f"YOLOX Enhanced comparison visualization done. Saved {vis_count}/{processed_count} images.")
             else: logger.warning("Image dir unreadable. Skipping YOLOX comparison vis.")
    else: logger.error("Could not determine optimal YOLOX combination.")
    logger.info("YOLOX Evaluation script finished.")

if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn',force=True); logger.info("Set multiprocessing start method 'spawn'")
    except RuntimeError as e: logger.warning(f"Start method set failed: {e}. Using default '{multiprocessing.get_start_method()}'.")
    except Exception as e: logger.warning(f"Error setting start method: {e}")
    main()
