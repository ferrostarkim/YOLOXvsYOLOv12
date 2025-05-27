#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Simplified YOLOX video processing script for performance testing and visualization.

import argparse
import os
import time
import cv2
import numpy as np
import sys
import torch
torch.backends.cudnn.benchmark = True # Added for potential performance improvement

from collections import deque
from loguru import logger
# from datetime import datetime # Not strictly needed for simplified version
# from pypylon import pylon # Comment out if not using Basler camera
# from basler_camera_set import Basler_Setup # Comment out if not using Basler camera

from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import VOC_CLASSES # Define class names directly
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess # vis is not used directly here

# Global variables
# frame_count = 0 # Moved to main processing loop as local variable

class MonitoringWindow:
    def __init__(self, window_name="YOLOX Inference Output"): # Modified window name
        self.is_open = True
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) # Make window resizable
        logger.info(f"Monitoring window '{self.window_name}' initialized.")

    def close(self):
        self.is_open = False
        cv2.destroyWindow(self.window_name)
        logger.info(f"Monitoring window '{self.window_name}' closed.")

    def show_frame(self, frame):
        if self.is_open:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF # Must have waitKey for imshow to work
            if key == ord('q'):
                self.close()
                return False # Signal to stop processing
        return self.is_open # Return whether the window is still open

# Simplified configuration, mainly using argparse now
# def load_simplified_config(args, actual_width, actual_height):
    # This function can be removed or further simplified if all settings come from args

def resize_frame_for_model(frame, target_width, target_height):
    # Simple cv2.resize for preprocessing, similar to yolov12_video_test.py
    # If using ValTransform later, it will handle its own resizing to model's test_size
    if target_width > 0 and target_height > 0:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return frame # Return original if target_width/height is not set

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Video Test - Simplified")
    parser.add_argument(
        "--mode", default="video", choices=['video', 'webcam'],
        help="Input mode: 'video' or 'webcam'"
    )
    parser.add_argument(
        "-f", "--exp_file",
        default="exps/default/yolox_s.py", # Provide a default YOLOX experiment file
        type=str,
        help="Path to YOLOX experiment file"
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="yolox_s.pth", # Provide a default checkpoint file
        type=str,
        help="Path to YOLOX checkpoint file (.pth)"
    )
    parser.add_argument(
        "--path",
        default="test_video.mp4", # Provide a default video path or webcam ID
        type=str,
        help="Path to video file or webcam ID (e.g., 0, 1)"
    )
    parser.add_argument(
        "--device", default="gpu", type=str, help="Device to use: 'cpu' or 'gpu'"
    )
    parser.add_argument(
        "--conf", default=0.25, type=float, help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--nms", default=0.45, type=float, help="NMS threshold for detection"
    )
    parser.add_argument(
        "--tsize", default=640, type=int,
        help="Target size for inference (square). Input frame will be resized if different. Set to 0 to use original frame size for model (if model supports dynamic shapes, otherwise ValTransform will resize to exp.test_size)."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 precision for inference"
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy YOLOX preprocessing (for older models)"
    )
    parser.add_argument(
        "--fuse", action="store_true", help="Fuse conv and bn layers for PyTorch model"
    )
    parser.add_argument(
        "--trt", action="store_true", help="Use TensorRT model for inference"
    )
    parser.add_argument(
        "--run_duration_sec", default=60, type=int,
        help="How long to run the video processing in seconds (0 for entire video)"
    )
    return parser

class Predictor:
    def __init__(
        self,
        model,
        exp,
        # Class names should be part of the experiment or model ideally
        cls_names=["baresho", "dokai", "ishi"], # Default, consider getting from exp.classes if available
        trt_file=None, # Path to the TRT engine file
        decoder=None, # YOLOX decoder
        device="gpu",
        fp16=False,
        legacy=False
    ):
        self.model = model
        self.cls_names = cls_names # Use provided or default class names
        if hasattr(exp, 'classes') and exp.classes: # If exp has class names, use them
             self.cls_names = exp.classes
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.test_size = exp.test_size # This is what ValTransform will use for preprocessing
        self.device = torch.device("cuda" if device == "gpu" else "cpu") # More robust device handling
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy) # Preprocessing transform

        if trt_file is not None:
            logger.info(f"Attempting to load TRT model from: {trt_file}")
            from torch2trt import TRTModule # Specific import for TRT
            model_trt = TRTModule()
            try:
                model_trt.load_state_dict(torch.load(trt_file, map_location=self.device))
                self.model = model_trt
                logger.info(f"Successfully loaded TRT model to {self.device}.")
                # Warmup TRT model
                logger.info("Warming up TRT model...")
                # Use a dummy image that matches the model's expected input size (exp.test_size)
                dummy_img_for_warmup = np.zeros((self.test_size[0], self.test_size[1], 3), dtype=np.uint8)
                img_tensor_warmup, _ = self.preproc(dummy_img_for_warmup, None, self.test_size)
                img_tensor_warmup = torch.from_numpy(img_tensor_warmup).unsqueeze(0).float().to(self.device)
                if self.fp16 and self.device.type == 'cuda':
                    img_tensor_warmup = img_tensor_warmup.half()

                for _ in range(10): # Perform a few warmup inferences
                    _ = self.model(img_tensor_warmup)
                logger.info("TRT model warmup complete.")
            except Exception as e:
                logger.error(f"Failed to load or warmup TRT model: {e}")
                raise # Re-raise exception if TRT loading fails

        self.model.to(self.device)
        if self.fp16 and self.device.type == 'cuda': # Check device type for half precision
            self.model.half()
        self.model.eval()

    def preprocess_image(self, img_bgr):
        # Preprocess the image using YOLOX's ValTransform
        # This will resize to self.test_size, normalize, etc.
        img_tensor, _ = self.preproc(img_bgr, None, self.test_size)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(self.device)
        if self.fp16 and self.device.type == 'cuda':
            img_tensor = img_tensor.half()
        return img_tensor

    def inference_raw_outputs(self, img_tensor):
        # Perform inference, return raw model outputs (before postprocessing)
        with torch.no_grad():
            outputs = self.model(img_tensor)
        return outputs

    def postprocess_outputs(self, raw_outputs, conf_threshold, nms_threshold):
        # Apply YOLOX postprocessing
        if self.decoder is not None: # For some YOLOX versions/TRT conversions
            raw_outputs = self.decoder(raw_outputs, dtype=raw_outputs.type())
        
        # outputs is a list, usually containing one tensor of detections
        processed_outputs_list = postprocess(
            raw_outputs, self.num_classes, conf_threshold,
            nms_threshold, class_agnostic=True # class_agnostic often True for YOLOX
        )
        return processed_outputs_list[0] # Return the tensor of detections, or None

    def visualize_detections(self, frame_to_draw_on, processed_detections_tensor, vis_conf_thresh):
        # Draw detections on the frame.
        # frame_to_draw_on: The frame (numpy array) on which to draw. This should be the frame
        #                   that corresponds to the detections (e.g., resized frame if model saw resized).
        # processed_detections_tensor: Tensor of detections after postprocessing.
        
        vis_frame = frame_to_draw_on.copy() # Draw on a copy

        if processed_detections_tensor is None or processed_detections_tensor.numel() == 0:
            return vis_frame # No detections to draw

        detections = processed_detections_tensor.cpu()
        
        # Scale boxes if the frame_to_draw_on is different from what the model saw
        # For simplicity, this example assumes ValTransform handled scaling and
        # the detections are relative to the self.test_size input.
        # If drawing on original frame, scaling is needed here.
        # For now, assume drawing on the frame that was (or would be) fed to model after preproc.

        bboxes = detections[:, 0:4]
        # If ValTransform resizes, bboxes are for that resized image.
        # If drawing on original, need to scale back.
        # ratio = min(frame_to_draw_on.shape[0] / self.test_size[0], frame_to_draw_on.shape[1] / self.test_size[1])
        # bboxes /= ratio # This would be needed if test_size is different from frame_to_draw_on.
                        # However, for simplicity, let's assume drawing on frame of self.test_size,
                        # or that detections are already scaled to frame_to_draw_on.
                        # For this lightweight version, we'll draw on the frame of target_inference_size.

        cls_ids = detections[:, 6]
        scores = detections[:, 4] * detections[:, 5] # YOLOX score: obj_conf * cls_conf

        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]

            if score < vis_conf_thresh: # Confidence threshold for visualization
                continue

            x0, y0, x1, y1 = map(int, box)
            class_name = self.cls_names[cls_id] if 0 <= cls_id < len(self.cls_names) else f"ID-{cls_id}"

            # Define colors (similar to yolov12_video_test)
            color = (0, 255, 0)  # Default Green
            if class_name.lower() == 'baresho':
                color = (255, 255, 255)  # White
            elif class_name.lower() == 'dokai':
                color = (0, 0, 255)  # Red
            elif class_name.lower() == 'ishi': # Ensure this class name matches
                color = (255, 0, 0)  # Blue (example)

            cv2.rectangle(vis_frame, (x0, y0), (x1, y1), color, 2)
            label = f"{class_name}: {score:.2f}"

            # Text drawing (similar to yolov12_video_test)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x0, y0 - text_h - baseline), (x0 + text_w, y0), color, -1)
            cv2.putText(vis_frame, label, (x0, y0 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1) # Black text

        return vis_frame

def video_processing_loop(predictor, args):
    cap = None
    # Basler camera related objects, uncomment if using
    # camera_pylon = None
    # converter = None

    # Initialize MonitoringWindow
    monitoring_window = MonitoringWindow()

    # Performance metrics deques
    frame_read_latencies = deque(maxlen=60)
    preprocess_latencies = deque(maxlen=60)
    inference_latencies = deque(maxlen=60) # Pure model inference
    postprocess_latencies = deque(maxlen=60)
    visualization_latencies = deque(maxlen=60)
    e2e_latencies = deque(maxlen=60) # End-to-end per frame

    # FPS calculation
    fps_update_freq_sec = 1.0
    last_fps_calc_time = time.perf_counter()
    frames_since_last_calc = 0
    current_display_fps = 0.0
    
    processed_frame_count = 0
    loop_start_time = time.perf_counter()

    try:
        # Determine target size for inference based on args.tsize
        # If args.tsize is 0, the model will see frames resized by ValTransform to exp.test_size.
        # If args.tsize > 0, frames are first resized to (tsize, tsize), then ValTransform processes this.
        # For simplicity, let's assume ValTransform always resizes to exp.test_size.
        # The frame given to predictor.visualize_detections should match the scale of detections.
        
        target_inference_width = predictor.test_size[1]
        target_inference_height = predictor.test_size[0]

        # --- Video/Webcam Initialization ---
        if args.mode == "webcam":
            cam_id = int(args.path) if args.path.isdigit() else args.path
            # if using Basler camera:
            # basler_setup = Basler_Setup('config.txt') # Assuming config.txt exists and is configured
            # camera_pylon = basler_setup.setup_camera()
            # camera_pylon.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            # converter = pylon.ImageFormatConverter()
            # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            # converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            # logger.info(f"Using Basler camera input.")
            # else use OpenCV for webcam:
            cap = cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_inference_width) # Try to set capture size
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_inference_height)
            if not cap.isOpened():
                raise IOError(f"Cannot open webcam {cam_id}")
            logger.info(f"Using OpenCV webcam {cam_id}.")
        else: # Video file
            if not os.path.exists(args.path):
                raise FileNotFoundError(f"Video file not found: {args.path}")
            cap = cv2.VideoCapture(args.path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {args.path}")
            logger.info(f"Processing video file: {args.path}")

        # --- Main Processing Loop ---
        while monitoring_window.is_open:
            if args.run_duration_sec > 0 and (time.perf_counter() - loop_start_time) > args.run_duration_sec:
                logger.info(f"Run duration of {args.run_duration_sec} seconds reached. Exiting.")
                break
            
            e2e_start_time = time.perf_counter()

            # 1. Read Frame
            read_start_time = time.perf_counter()
            frame_bgr = None
            if args.mode == "webcam":
                # if camera_pylon and camera_pylon.IsGrabbing():
                #     grab_result = camera_pylon.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                #     if grab_result.GrabSucceeded():
                #         frame_bgr = converter.Convert(grab_result).GetArray()
                #     grab_result.Release()
                # else: # OpenCV webcam
                ret, frame_bgr = cap.read()
                if not ret:
                    logger.info("Webcam stream ended or failed to grab frame.")
                    break
            else: # Video file
                ret, frame_bgr = cap.read()
                if not ret:
                    logger.info("End of video file.")
                    break
            read_end_time = time.perf_counter()
            frame_read_latencies.append((read_end_time - read_start_time) * 1000)

            if frame_bgr is None:
                logger.warning("Frame is None, skipping.")
                continue

            # (Optional) Resize original frame if a specific display/inference size is set by args.tsize
            # This frame_for_inference will be what the model sees after ValTransform.
            # ValTransform itself resizes to predictor.test_size.
            # If args.tsize is for initial resize before ValTransform:
            current_frame_for_inference = frame_bgr
            if args.tsize > 0: # If user specified a tsize for initial resize
                 # This resize is before ValTransform, ValTransform will do its own resize to exp.test_size
                 # For simplicity and to match yolov12_video_test, let's make tsize the target for display
                 # And assume ValTransform handles model input size correctly.
                 # The frame sent to visualize_detections should be what ValTransform processed.
                 # Let's assume predictor.test_size is the size ValTransform makes it.
                 # We'll resize the display frame to target_inference_width/height for consistency.
                 display_frame_resized = resize_frame_for_model(frame_bgr, target_inference_width, target_inference_height)
                 current_frame_for_model_input = display_frame_resized # This is what ValTransform will process
            else: # Use original frame (ValTransform will resize it)
                 display_frame_resized = frame_bgr.copy() # Draw on a copy of original if not resized
                 current_frame_for_model_input = frame_bgr


            # 2. Preprocess Frame for Model
            preprocess_start_time = time.perf_counter()
            img_tensor = predictor.preprocess_image(current_frame_for_model_input) # ValTransform applied here
            preprocess_end_time = time.perf_counter()
            preprocess_latencies.append((preprocess_end_time - preprocess_start_time) * 1000)

            # 3. Inference
            inference_start_time = time.perf_counter()
            raw_model_outputs = predictor.inference_raw_outputs(img_tensor)
            inference_end_time = time.perf_counter()
            inference_latencies.append((inference_end_time - inference_start_time) * 1000)

            # 4. Postprocess
            postprocess_start_time = time.perf_counter()
            # Use args.conf and args.nms for postprocessing thresholds
            processed_detections = predictor.postprocess_outputs(raw_model_outputs, args.conf, args.nms)
            postprocess_end_time = time.perf_counter()
            postprocess_latencies.append((postprocess_end_time - postprocess_start_time) * 1000)
            
            # 5. Visualize Detections
            # Detections are for the image size that ValTransform created (predictor.test_size)
            # So, we should visualize on a frame of that size.
            # 'display_frame_resized' is already at target_inference_width/height which should be predictor.test_size
            vis_start_time = time.perf_counter()
            # The frame passed to visualize_detections should be the one preprocessed by ValTransform,
            # or a frame of the same dimensions (predictor.test_size) for correct bbox coordinates.
            # `current_frame_for_model_input` was what went into preproc. If it was resized to predictor.test_size
            # by preproc, then detections are for that size.
            # For lightweight version, draw on 'display_frame_resized' (which is at target_inference_size)
            # Ensure predictor.test_size matches target_inference_size if tsize=0
            
            # If args.tsize was 0, display_frame_resized is original. We need to resize it to predictor.test_size for drawing
            # Or, scale detections back to original frame.
            # Let's draw on display_frame_resized, assuming it matches predictor.test_size or detections are scaled.
            # The detections from YOLOX postprocess are usually for the input image to the network (after ValTransform).

            frame_to_visualize_on = current_frame_for_model_input # This frame was input to ValTransform
            if args.tsize > 0 : # if frame_bgr was resized to args.tsize before ValTransform
                # ValTransform further resized it to predictor.test_size.
                # We need a frame of predictor.test_size to draw on for correct bbox coords.
                # Let's assume current_frame_for_model_input is already at predictor.test_size (implicitly by ValTransform)
                # or that predictor.visualize_detections handles scaling if needed.
                # To simplify: assume ValTransform resizes input to predictor.test_size
                # and detections are for that size. We need a frame of that size to draw on.
                # If current_frame_for_model_input was original, it will be resized by ValTransform.
                # The simplest is to use a black image of predictor.test_size and draw on it if needed,
                # or ensure display_frame_resized is actually predictor.test_size.
                # For now, let's use display_frame_resized which is target_inference_width/height.
                # This assumes target_inference_width/height IS predictor.test_size for visualization.

                # Simplification: Visualize on the frame that was fed to `predictor.preprocess_image`
                # This frame is `current_frame_for_model_input`.
                # The `predictor.visualize_detections` should expect detections corresponding to this frame's scale
                # *after* `ValTransform` has processed it.
                # Detections from `postprocess` are typically scaled to the `test_size` image.
                # So, `visualize_detections` needs an image of `test_size` to draw on.
                
                # Let's make a blank image of test_size if current_frame_for_model_input isn't already it.
                # OR, resize current_frame_for_model_input to test_size for drawing.
                vis_base_frame = cv2.resize(current_frame_for_model_input, (predictor.test_size[1], predictor.test_size[0]))

            else: # Original size was used as input to preproc
                vis_base_frame = cv2.resize(current_frame_for_model_input, (predictor.test_size[1], predictor.test_size[0]))


            result_frame = predictor.visualize_detections(
                vis_base_frame, # Draw on a frame that matches model's input dimensions
                processed_detections,
                args.conf # Use args.conf for visualization threshold too
            )
            vis_end_time = time.perf_counter()
            visualization_latencies.append((vis_end_time - vis_start_time) * 1000)

            e2e_end_time = time.perf_counter()
            e2e_latencies.append((e2e_end_time - e2e_start_time) * 1000)

            # Display the frame
            if not monitoring_window.show_frame(result_frame):
                break # Exit if 'q' is pressed or window closed

            processed_frame_count += 1
            frames_since_last_fps_update +=1

            # Calculate and print FPS periodically
            current_time = time.perf_counter()
            if (current_time - last_fps_calc_time) >= fps_update_freq_sec:
                current_display_fps = frames_since_last_fps_update / (current_time - last_fps_calc_time)
                avg_e2e_ms = sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0
                avg_read_ms = sum(frame_read_latencies)/len(frame_read_latencies) if frame_read_latencies else 0
                avg_pre_ms = sum(preprocess_latencies)/len(preprocess_latencies) if preprocess_latencies else 0
                avg_inf_ms = sum(inference_latencies)/len(inference_latencies) if inference_latencies else 0
                avg_post_ms = sum(postprocess_latencies)/len(postprocess_latencies) if postprocess_latencies else 0
                avg_vis_ms = sum(visualization_latencies)/len(visualization_latencies) if visualization_latencies else 0

                print(f"\rFPS: {current_display_fps:.2f} | "
                      f"Frames: {processed_frame_count} | "
                      f"E2E: {avg_e2e_ms:.1f}ms (Read:{avg_read_ms:.1f} Prep:{avg_pre_ms:.1f} Inf:{avg_inf_ms:.1f} Post:{avg_post_ms:.1f} Vis:{avg_vis_ms:.1f})",
                      end="", flush=True)
                
                frames_since_last_fps_update = 0
                last_fps_calc_time = current_time

    except Exception as e:
        logger.error(f"Error during video processing loop: {e}", exc_info=True)
    finally:
        logger.info("\nCleaning up resources...")
        if cap is not None:
            cap.release()
        # if camera_pylon is not None and camera_pylon.IsGrabbing(): # Basler cleanup
            # camera_pylon.StopGrabbing()
        # if camera_pylon is not None and camera_pylon.IsOpen():
            # camera_pylon.Close()
        monitoring_window.close() # Ensures cv2.destroyAllWindows() is called
        print("\nProcessing finished.")


def main(exp, args):
    if not hasattr(exp, 'output_dir') or exp.output_dir is None:
         exp.output_dir = "./yolox_exp_outputs" # Default output if not in exp

    if not args.experiment_name: # From original YOLOX demo
        args.experiment_name = exp.exp_name

    # Output directory for experiment (not really used in this simplified version)
    # file_name = os.path.join(exp.output_dir, args.experiment_name)
    # os.makedirs(file_name, exist_ok=True) # Create if it doesn't exist

    logger.info(f"Arguments: {args}")

    # Set experiment parameters from args
    if args.conf is not None:
        exp.test_conf = args.conf # Used by Predictor during postprocess
    if args.nms is not None:
        exp.nmsthre = args.nms # Used by Predictor during postprocess
    
    # test_size for ValTransform is taken from exp.test_size
    # args.tsize can be used for an initial resize before ValTransform if needed,
    # or to set the display window size.
    # For this script, exp.test_size dictates model input size.
    if args.tsize is not None and args.tsize > 0:
        logger.info(f"Note: args.tsize ({args.tsize}) is for initial resize/display. Model sees exp.test_size ({exp.test_size}).")
        # If you want args.tsize to override exp.test_size for the model:
        # exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model() # Get base YOLOX model structure
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size))) # Can be verbose

    if args.device == "gpu":
        model.cuda() # Move to GPU before potential FP16 conversion or TRT loading
        if args.fp16:
            model.half()
    model.eval()

    trt_file_path = None
    decoder_for_trt = None # Decoder might be needed for TRT model's output

    if args.trt:
        if args.device != "gpu":
            logger.warning("TensorRT is typically used on GPU. Setting device to GPU.")
            args.device = "gpu" # Force GPU for TRT
            model.cuda() # Ensure model is on GPU for TRT conversion/loading
            if args.fp16: model.half()

        assert not args.fuse, "TensorRT model does not support model fusing (--fuse)."
        
        # Define path for TRT engine file (original YOLOX demo convention)
        # This path might need adjustment based on how TRT models are saved/named
        # For example, it might be args.ckpt if args.ckpt is already a TRT file.
        # Assuming args.ckpt is PyTorch, and trt_file is derived or a separate arg.
        # For this script, let's assume args.ckpt can be a TRT model if --trt is set.
        # This simplifies things: if --trt, args.ckpt IS the TRT model path.
        trt_file_path = args.ckpt # Assume --ckpt points to the TRT model file when --trt is active
        
        if not os.path.exists(trt_file_path):
            logger.error(f"TensorRT model file not found: {trt_file_path}. Please ensure --ckpt points to a valid TRT model when --trt is used.")
            sys.exit(1)
        
        # For some YOLOX versions, the head's decoding needs to be handled differently with TRT
        if hasattr(model, 'head') and hasattr(model.head, 'decode_in_inference'):
            model.head.decode_in_inference = False # Disable internal decoding for TRT
            decoder_for_trt = model.head.decode_outputs # Get the decoder function
        logger.info("Using TensorRT model for inference.")
    else: # Standard PyTorch checkpoint
        if args.ckpt is None:
            # Default ckpt path logic from original YOLOX demo (might not be needed here)
            # ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            logger.error("PyTorch checkpoint file (--ckpt) must be provided if not using --trt.")
            sys.exit(1)
        else:
            ckpt_file = args.ckpt

        if not os.path.exists(ckpt_file):
            logger.error(f"Checkpoint file not found: {ckpt_file}")
            sys.exit(1)
            
        logger.info(f"Loading PyTorch checkpoint from: {ckpt_file}")
        try:
            loc = "cuda" if args.device == "gpu" else "cpu"
            ckpt_loaded = torch.load(ckpt_file, map_location=loc)
            # Load model state dict
            if "model" in ckpt_loaded:
                model.load_state_dict(ckpt_loaded["model"])
            else: # If it's just the state_dict
                model.load_state_dict(ckpt_loaded)
            logger.info("Checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            sys.exit(1)
        
        if args.fuse:
            logger.info("Fusing model (Conv+BN)...")
            model = fuse_model(model) # Fuse layers if requested

    # Initialize Predictor
    # Class names can be derived from exp if available, e.g., exp.classes
    # For simplicity, using a fixed list or allowing override via args could be an option.
    # Here, using a default and checking exp.
    cls_names_from_exp = VOC_CLASSES # Default fallback from YOLOX
    if hasattr(exp, 'dataset') and hasattr(exp.dataset, 'class_names'):
        cls_names_from_exp = exp.dataset.class_names
    elif hasattr(exp, 'class_names'): # Some exps might define it directly
        cls_names_from_exp = exp.class_names


    predictor = Predictor(
        model, exp,
        cls_names=cls_names_from_exp,
        trt_file=trt_file_path if args.trt else None, # Pass TRT file path if --trt
        decoder=decoder_for_trt if args.trt else None, # Pass decoder if using TRT
        device=args.device,
        fp16=args.fp16,
        legacy=args.legacy
    )

    # Load simplified config (not using external file for this version)
    # Actual width/height for config will be determined after opening video/camera
    # For now, pass None, and it will be set in video_processing_loop
    # config = load_simplified_config(args, actual_width=640, actual_height=480) # Placeholder, will be updated

    # Start video processing
    video_processing_loop(predictor, args) # Removed config dependency for now

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name) # args.name is for exp, not model name in this context
    
    # Setup logger
    logger.remove() # Remove default logger
    log_level = "INFO" # Can be configured
    logger.add(sys.stderr, level=log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    main(exp, args)