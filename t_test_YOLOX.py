#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX Statistical Evaluation Script - Strict Mode

import argparse
import os
import time
import cv2
import numpy as np
import torch
import logging
import sys
from collections import defaultdict
from loguru import logger
from datetime import datetime
from tqdm import tqdm

# Import YOLOX modules
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

# Manually define VOC_CLASSES based on the project
VOC_CLASSES = ['baresho', 'dokai', 'ishi']

def make_parser():
    parser = argparse.ArgumentParser("YOLOX VOC Statistical Evaluation")
    parser.add_argument("-expn", "--experiment-name", 
                       type=str, 
                       default=None)
    parser.add_argument("-n", "--name", 
                       type=str, 
                       default="yolox-s", 
                       help="model name")
    parser.add_argument("-f", "--exp_file", 
                       default="exps/yolox_voc_s.py", 
                       type=str, 
                       help="experiment description file")
    parser.add_argument("-c", "--ckpt", 
                       default="checkpoints/yolox_voc_s.pth", 
                       type=str, 
                       help="checkpoint file")
    parser.add_argument("--device", 
                       default="gpu", 
                       type=str, 
                       help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", 
                       default=0.5, 
                       type=float, 
                       help="test conf")
    parser.add_argument("--nms", 
                       default=0.3, 
                       type=float, 
                       help="test nms threshold")
    parser.add_argument("--tsize", 
                       default=None, 
                       type=int, 
                       help="test img size")
    parser.add_argument("--fp16", 
                       dest="fp16", 
                       default=False, 
                       action="store_true",
                       help="Adopting mix precision evaluating")
    parser.add_argument("--legacy", 
                       dest="legacy", 
                       default=False, 
                       action="store_true",
                       help="To be compatible with older versions")
    parser.add_argument("--fuse", 
                       dest="fuse", 
                       default=False, 
                       action="store_true",
                       help="Fuse conv and bn for testing")
    parser.add_argument("--trt", 
                       dest="trt", 
                       default=False, 
                       action="store_true",
                       help="Using TensorRT model for testing")
    parser.add_argument("--by_class", 
                       dest="by_class", 
                       default=False, 
                       action="store_true",
                       help="Evaluate by class")
    parser.add_argument("--by_size", 
                       dest="by_size", 
                       default=False, 
                       action="store_true",
                       help="Evaluate by object size")
    parser.add_argument("--num_runs", 
                       type=int, 
                       default=1,
                       help="Number of evaluation runs for statistical analysis")
    parser.add_argument("--visual", 
                       dest="visual", 
                       default=False, 
                       action="store_true",
                       help="Visualize detection results")
    parser.add_argument("--output_dir", 
                       type=str, 
                       default="eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--data_dir", 
                       type=str, 
                       default="datasets/VOCdevkit",
                       help="VOC dataset directory")
    parser.add_argument("--strict", 
                       dest="strict", 
                       default=False, 
                       action="store_true",
                       help="Use strict loading mode for checkpoint")
    
    return parser

class SizeAnalyzer:
    """Helper class for object size-based evaluation"""
    
    SIZE_CATEGORIES = {
        "very_small": (0, 3000),
        "small": (3000, 6000),
        "medium": (6000, 9000),
        "medium_large": (9000, 12000),
        "large": (12000, 15000),
        "x_large": (15000, 18000),
        "xx_large": (18000, 21000),
        "xxx_large": (21000, float('inf')),
    }
    
    @staticmethod
    def get_size_category(bbox):
        """Calculate size category based on bounding box area"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        for category, (min_area, max_area) in SizeAnalyzer.SIZE_CATEGORIES.items():
            if min_area <= area < max_area:
                return category
        
        return "xxx_large"  # Default fallback

class Predictor:
    """YOLOX model predictor class"""
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        
        # TensorRT model loading
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)  # Warmup
            self.model = model_trt
    
    def inference(self, img):
        """Run inference on an image"""
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.debug("Inference time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        """Visualize detection results"""
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, []
        
        output = output.cpu()

        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # Create visualization using OpenCV
        vis_img = img.copy()
        
        for i, (box, score, cl) in enumerate(zip(bboxes, scores, cls)):
            x0, y0, x1, y1 = [int(i) for i in box]
            cls_id = int(cl)
            
            if cls_id < len(self.cls_names):
                cls_name = self.cls_names[cls_id]
            else:
                cls_name = f"unknown-{cls_id}"
                
            # Set color based on class name
            if cls_name.lower() == 'baresho':
                color = (255, 255, 255)  # White for Baresho
            elif cls_name.lower() == 'dokai':
                color = (0, 0, 255)  # Red for Dokai
            elif cls_name.lower() == 'ishi':
                color = (255, 0, 0)  # Blue for Ishi
            else:
                color = (0, 255, 0)  # Green for other classes
                
            # Draw rectangle
            cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
            
            # Draw text
            text = f"{cls_name}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            cv2.rectangle(vis_img, (x0, y0 - text_size[1] - 4), (x0 + text_size[0], y0), color, -1)
            cv2.putText(vis_img, text, (x0, y0 - 2), font, 0.5, (255, 255, 255), 1)
            
        return vis_img, bboxes, scores, cls

def get_voc_dataset(data_dir, img_size, legacy=False):
    """Load VOC dataset images for evaluation"""
    from glob import glob
    
    # Check if VOC dataset exists
    voc_path = os.path.join(data_dir, 'VOC2007', 'JPEGImages')
    if not os.path.exists(voc_path):
        logger.error(f"VOC dataset not found at {voc_path}")
        raise FileNotFoundError(f"VOC dataset not found at {voc_path}")
    
    # Get test images
    image_paths = sorted(glob(os.path.join(voc_path, '*.jpg')))
    
    if not image_paths:
        logger.error(f"No images found in {voc_path}")
        raise ValueError(f"No images found in {voc_path}")
    
    logger.info(f"Found {len(image_paths)} test images")
    
    # Get annotations
    anno_path = os.path.join(data_dir, 'VOC2007', 'Annotations')
    if not os.path.exists(anno_path):
        logger.warning(f"Annotations directory not found at {anno_path}")
    
    return image_paths

def evaluate_model(predictor, image_paths, args):
    """Evaluate model performance on a dataset"""
    # Initialize metrics
    results = defaultdict(list)
    size_results = defaultdict(lambda: defaultdict(list))
    inference_times = []
    
    # Create output directories
    output_dir = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.visual:
        visual_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visual_dir, exist_ok=True)
    
    # Process each image
    logger.info(f"Running evaluation on {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        # Get image ID (filename without extension)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # Run inference
        t0 = time.time()
        outputs, img_info = predictor.inference(img_path)
        inference_time = time.time() - t0
        inference_times.append(inference_time)
        
        # Process detection results
        if outputs[0] is not None:
            output = outputs[0].cpu()
            ratio = img_info["ratio"]
            
            bboxes = output[:, 0:4] / ratio
            cls_ids = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            
            # Store results for each detection
            for idx, (bbox, cls_id, score) in enumerate(zip(bboxes, cls_ids, scores)):
                cls_id = int(cls_id)
                if cls_id < len(predictor.cls_names):
                    cls_name = predictor.cls_names[cls_id]
                else:
                    cls_name = f"unknown-{cls_id}"
                
                x0, y0, x1, y1 = [float(i) for i in bbox]
                
                # Store result
                detection_result = {
                    "bbox": [x0, y0, x1, y1],
                    "score": float(score),
                    "img_id": img_id,
                    "inference_time": inference_time,
                }
                
                results[cls_name].append(detection_result)
                
                # Size-based evaluation
                if args.by_size:
                    size_category = SizeAnalyzer.get_size_category([x0, y0, x1, y1])
                    size_results[cls_name][size_category].append(detection_result)
        
        # Visualize results if requested
        if args.visual and outputs[0] is not None:
            vis_img, _, _, _ = predictor.visual(outputs[0], img_info)
            save_path = os.path.join(visual_dir, f"{os.path.basename(img_path)}")
            cv2.imwrite(save_path, vis_img)
    
    # Calculate average FPS
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Save results
    logger.info(f"Saving evaluation results to {output_dir}")
    
    # Overall metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Average inference time: {avg_inference_time:.4f} seconds\n")
        f.write(f"FPS: {fps:.2f}\n\n")
        
        f.write("Detection counts by class:\n")
        for cls_name, detections in results.items():
            f.write(f"{cls_name}: {len(detections)} detections\n")
    
    # Class-specific analysis
    if args.by_class:
        for cls_name, detections in results.items():
            with open(os.path.join(output_dir, f"{cls_name}_analysis.txt"), "w") as f:
                total_detections = len(detections)
                if total_detections > 0:
                    avg_score = sum(d["score"] for d in detections) / total_detections
                    f.write(f"Class: {cls_name}\n")
                    f.write(f"Total detections: {total_detections}\n")
                    f.write(f"Average confidence score: {avg_score:.4f}\n")
    
    # Size-specific analysis
    if args.by_size:
        with open(os.path.join(output_dir, "size_analysis.txt"), "w") as f:
            for cls_name, size_cats in size_results.items():
                f.write(f"\nClass: {cls_name}\n")
                f.write("-" * 40 + "\n")
                for size_cat, detections in size_cats.items():
                    if detections:
                        total = len(detections)
                        avg_score = sum(d["score"] for d in detections) / total
                        f.write(f"Size: {size_cat}\n")
                        f.write(f"  Total detections: {total}\n")
                        f.write(f"  Average confidence: {avg_score:.4f}\n")
    
    return {
        "fps": fps,
        "inference_time": avg_inference_time,
        "detection_counts": {cls: len(dets) for cls, dets in results.items()}
    }

def run_statistical_evaluation(args):
    """Run multiple evaluations for statistical analysis"""
    # Load experiment
    exp = get_exp(args.exp_file, args.name)
    
    # Update experimental parameters if provided
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VOC dataset
    image_paths = get_voc_dataset(args.data_dir, exp.test_size, args.legacy)
    
    # Variables to store statistical results
    fps_values = []
    inference_times = []
    detection_counts = defaultdict(list)
    
    # Run multiple evaluations
    num_runs = args.num_runs
    logger.info(f"Running {num_runs} evaluation(s) for statistical analysis...")
    
    for run in range(num_runs):
        logger.info(f"Starting evaluation run {run+1}/{num_runs}")
        
        # Load model
        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        
        # Move model to device
        if args.device == "gpu":
            model.cuda()
            if args.fp16:
                model.half()
        model.eval()
        
        # Setup TensorRT if enabled
        decoder = None
        trt_file = None
        if args.trt:
            trt_file = os.path.join(exp.output_dir, "model_trt.pth")
            assert os.path.exists(trt_file), "TensorRT model not found!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
        else:
            # Load checkpoint
            ckpt_file = args.ckpt
            logger.info(f"Loading checkpoint: {ckpt_file}")
            
            try:
                ckpt = torch.load(ckpt_file, map_location="cpu")
                
                # Handle the case where checkpoint format doesn't match model structure
                if not args.strict:
                    logger.info("Loading checkpoint in non-strict mode...")
                    # Filter out unexpected keys
                    state_dict = ckpt["model"]
                    model_state_dict = model.state_dict()
                    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
                    
                    # Check if required keys are missing
                    missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
                    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
                    
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")
                    
                    model.load_state_dict(filtered_state_dict, strict=False)
                else:
                    model.load_state_dict(ckpt["model"])
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.info("Trying direct model load without 'model' key...")
                try:
                    # Some checkpoints might be saved without the 'model' key
                    state_dict = torch.load(ckpt_file, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e2:
                    logger.error(f"Failed with direct load as well: {e2}")
                    raise RuntimeError("Could not load checkpoint. Try using --strict False")
            
            # Fuse model if requested
            if args.fuse:
                logger.info("Fusing model...")
                model = fuse_model(model)
        
        # Create predictor
        predictor = Predictor(
            model, 
            exp, 
            VOC_CLASSES, 
            trt_file, 
            decoder, 
            args.device, 
            args.fp16, 
            args.legacy
        )
        
        # Run evaluation
        run_result = evaluate_model(predictor, image_paths, args)
        
        # Store results
        fps_values.append(run_result["fps"])
        inference_times.append(run_result["inference_time"])
        
        for cls, count in run_result["detection_counts"].items():
            detection_counts[cls].append(count)
    
    # Statistical analysis
    logger.info("Computing statistical results...")
    
    # Create statistical results
    mean_fps = np.mean(fps_values)
    std_fps = np.std(fps_values)
    
    mean_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    
    # Write statistical results
    with open(os.path.join(args.output_dir, "statistical_results.txt"), "w") as f:
        f.write(f"Statistical Analysis ({num_runs} runs)\n")
        f.write("-" * 40 + "\n\n")
        
        f.write(f"FPS: {mean_fps:.2f} ± {std_fps:.2f}\n")
        f.write(f"Inference time (s): {mean_inference:.4f} ± {std_inference:.4f}\n\n")
        
        f.write("Detection counts by class:\n")
        for cls, counts in detection_counts.items():
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            f.write(f"{cls}: {mean_count:.1f} ± {std_count:.1f}\n")
        
        f.write("\nRaw data:\n")
        f.write(f"FPS values: {[round(x, 2) for x in fps_values]}\n")
        f.write(f"Inference times: {[round(x, 4) for x in inference_times]}\n")
        
        # T-test if we have enough runs
        if num_runs >= 5:
            try:
                from scipy import stats
                
                f.write("\nT-test for FPS (H0: mean=0):\n")
                t_stat, p_value = stats.ttest_1samp(fps_values, 0)
                f.write(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.8f}\n")
                
                # If we have class counts, run t-tests for each class
                f.write("\nT-test for detection counts by class (H0: mean=0):\n")
                for cls, counts in detection_counts.items():
                    t_stat, p_value = stats.ttest_1samp(counts, 0)
                    f.write(f"{cls}: t={t_stat:.4f}, p={p_value:.8f}\n")
                    
            except ImportError:
                f.write("\nNote: scipy not available for t-test analysis\n")
    
    logger.info(f"Statistical results saved to {os.path.join(args.output_dir, 'statistical_results.txt')}")
    return mean_fps

def main():
    # Parse arguments
    args = make_parser().parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")  # DEBUG에서 INFO로 변경
logger.add(f"{args.output_dir}_log.txt", rotation="10 MB", level="INFO")  # 로그 파일도 INFO 레벨로
    
    logger.info("Args: {}".format(args))
    
    # Run statistical evaluation
    mean_fps = run_statistical_evaluation(args)
    
    logger.info(f"Evaluation completed successfully. Average FPS: {mean_fps:.2f}")

if __name__ == "__main__":
    main()