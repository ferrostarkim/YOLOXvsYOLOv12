# -*- coding: utf-8 -*-
# yolov12_video_test_modified.py

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
import subprocess
import csv
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import shutil
from multiprocessing import Process, Queue, Event, set_start_method, get_start_method
from queue import Empty, Full
from ultralytics import YOLO
import torch
import gc

# --- Configuration (Hardcoded Defaults) ---
VIDEO_SOURCE = 'live_2km_base_1280x960.avi'
YOLOV12_ENGINE_PATH = 'yolov12s.engine'
CONF_THRESHOLD = 0.25
CLASS_NAMES = {0: 'baresho', 1: 'dokai', 2: 'ishi'} # Example, adjust if needed
RUN_DURATION_SEC = 60
DEVICE = 'cuda:0'
USE_TRT = True # Keep True as we load .engine
OUTPUT_DIR_BASE = 'yolov12_results'
SUMMARY_CSV = 'yolov12_summary.csv'
DETAILED_LOG_DIR_NAME = 'detailed_logs'
PLOT_DIR_NAME = 'plots'
RESIZE_DIM = (640, 640) # Define resize dimension

# --- Helper Functions ---

def load_yolo_model_trt_only(engine_path, device_arg):
    """Loads the YOLOv12 TRT engine and warms it up."""
    engine_path = Path(engine_path)
    if not engine_path.exists(): raise FileNotFoundError(f"TRT engine not found: {engine_path}")
    print(f"Loading YOLOv12 TRT engine from: {engine_path}"); load_start = time.perf_counter()
    try:
        model = YOLO(engine_path, task='detect') # Explicitly set task
        class_names = model.names; print("Warming up TRT engine...")
        dummy_image = np.zeros(RESIZE_DIM + (3,), dtype=np.uint8) # Use resize dim for warmup
        for _ in range(5): model.predict(dummy_image, verbose=False, device=device_arg)
        print("Warmup finished."); load_end = time.perf_counter()
        print(f"YOLOv12 TRT engine loaded/warmed up in {(load_end - load_start)*1000:.1f} ms.")
        return model, class_names
    except Exception as e: print(f"Error loading model: {e}"); raise

def log_detections_yolov12(frame_id, boxes_data, log_writer):
    """Logs detected bounding boxes to a CSV file."""
    if boxes_data is not None and hasattr(boxes_data, 'xyxy') and log_writer is not None:
        try:
            if hasattr(boxes_data, 'conf') and hasattr(boxes_data, 'cls'):
                for i in range(len(boxes_data.xyxy)):
                    box = boxes_data.xyxy[i]; conf = boxes_data.conf[i]; cls_id = int(boxes_data.cls[i])
                    x1, y1, x2, y2 = map(int, box)
                    log_writer.writerow([frame_id, cls_id, f"{conf:.4f}", x1, y1, x2, y2])
            else: print(f"Warning: boxes_data object lacks 'conf' or 'cls' attributes for frame {frame_id}.")
        except IndexError: print(f"Warning: Index error while accessing detection data for frame {frame_id}.")
        except Exception as e: print(f"Warning: Error logging YOLOv12 detection for frame {frame_id}: {e}")

def draw_detections_yolov12(frame, boxes_data, class_names, baseline_y):
    """Draws bounding boxes on the frame."""
    drawn_frame = frame.copy()
    cv2.line(drawn_frame, (0, baseline_y), (drawn_frame.shape[1], baseline_y), (0, 255, 255), 2)
    if boxes_data is not None and hasattr(boxes_data, 'xyxy') and len(boxes_data.xyxy) > 0:
        if hasattr(boxes_data, 'conf') and hasattr(boxes_data, 'cls'):
            for i in range(len(boxes_data.xyxy)):
                try:
                    box = boxes_data.xyxy[i]; conf = boxes_data.conf[i]; cls_id = int(boxes_data.cls[i])
                    class_name = class_names.get(cls_id, f"ID-{cls_id}"); x1, y1, x2, y2 = map(int, box)
                    crosses_baseline = (y1 <= baseline_y <= y2); color = (0, 0, 255) if crosses_baseline else (0, 255, 0)
                    cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2); label = f"{class_name} {conf:.2f}"
                    (w, h), baseline_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(drawn_frame, (x1, y1 - h - baseline_text), (x1 + w, y1), color, -1)
                    cv2.putText(drawn_frame, label, (x1, y1 - baseline_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                except IndexError: print(f"Warning: Index error while drawing detection {i}."); continue
                except Exception as e: print(f"Warning: Error drawing detection {i}: {e}"); continue
        else: print("Warning: Cannot draw detections, boxes_data object lacks 'conf' or 'cls' attributes.")
    return drawn_frame

# --- Plot Results Function (Simplified) ---
def plot_results(latencies, output_dir, mode_name):
    """Generates plots for latency."""
    print("Generating plots...")
    plot_dir = Path(output_dir) / PLOT_DIR_NAME; plot_dir.mkdir(parents=True, exist_ok=True)
    has_latency = any(latencies.values())
    if not has_latency: 
        print("No latency data available for plotting.")
        return
    
    # Inference Latency Histogram
    valid_infer_latencies_hist = list(latencies['infer'])
    plt.figure(figsize=(10, 6))
    if valid_infer_latencies_hist:
        avg_lat = np.mean(valid_infer_latencies_hist)
        plt.hist(valid_infer_latencies_hist, bins=30, alpha=0.7, color='green')
        plt.axvline(avg_lat, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_lat:.1f} ms')
        plt.xlabel("Inference Latency (ms)")
        plt.ylabel("Frequency")
        plt.title(f"Inference Latency Distribution ({mode_name})")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / f"{mode_name}_inference_latency_hist.png")
        print(f"Saved latency histogram plot to: {plot_dir / f'{mode_name}_inference_latency_hist.png'}")
    else: 
        print("No inference latency data in deque to plot histogram.")
    plt.close()
    print(f"All plots saved to: {plot_dir}")

# --- YOLOv12 Inference Worker ---
def inference_process_worker(engine_path, conf_threshold, device_arg, input_queue: Queue, output_queue: Queue, stop_event: Event):
    """Worker process for running YOLO inference."""
    import torch; import time; from queue import Empty, Full
    print(f"[InferProc] Worker started (PID: {os.getpid()})"); model = None; class_names = None
    try:
        model, class_names = load_yolo_model_trt_only(engine_path, device_arg); output_queue.put(("CLASS_NAMES", class_names))
    except Exception as e:
        print(f"[InferProc] Error loading model: {e}")
        try: output_queue.put(("ERROR", str(e)))
        except Exception as q_err: print(f"[InferProc] Error sending model load error to queue: {q_err}")
        return
    while not stop_event.is_set():
        try:
            frame_data = input_queue.get(timeout=0.5)
            if frame_data is None: break
            frame_id, frame_resized_for_inf = frame_data
            infer_start = time.perf_counter()
            results = model.predict(frame_resized_for_inf, conf=conf_threshold, verbose=False, device=device_arg)
            infer_end = time.perf_counter(); infer_time_ms = (infer_end - infer_start) * 1000
            boxes_data = results[0].boxes.cpu().numpy() if results and results[0].boxes is not None and len(results[0].boxes) > 0 else None
            try: output_queue.put((frame_id, boxes_data, infer_time_ms), timeout=0.5)
            except Full: print("[InferProc] Warning: Output queue full. Discarding result."); time.sleep(0.01); continue
        except Empty: continue
        except KeyboardInterrupt: print("[InferProc] KeyboardInterrupt caught in worker loop. Exiting gracefully."); break
        except (BrokenPipeError, EOFError): print("[InferProc] Communication pipe broken. Exiting."); break
        except Exception as e:
            if not isinstance(e, Full): print(f"[InferProc] Inference loop error: {type(e).__name__}: {e}")
            continue
    print("[InferProc] Worker loop exited.")
    if model is not None:
        try: del model; model = None
        except Exception as del_err: print(f"[InferProc] Error deleting model object: {del_err}")
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache(); print("[InferProc] Cleared CUDA cache.")
        except Exception as cuda_err: print(f"[InferProc] Error clearing CUDA cache: {cuda_err}")
    print("[InferProc] Worker finished cleanup and exiting.")


# --- Main YOLOv12 Benchmark Function ---
def run_yolov12_benchmark():
    """Runs the YOLOv12 benchmark using 2 processes with serial IPC flow."""
    mode_name = "yolov12_trt_benchmark"
    print(f"Starting YOLOv12 Benchmark ({mode_name})...")
    print("Processing Mode: SERIALIZED IPC")

    # Define variables in the outer scope for the finally block
    inference_proc = None; frame_input_queue = None; results_output_queue = None
    stop_event = None; cap = None; det_log_file = None; det_log_writer = None
    class_names = None
    latencies = defaultdict(lambda: deque(maxlen=int(30 * 5)))

    try: # Wrap entire function logic for robust cleanup
        # Setup output directories
        current_output_dir = Path(OUTPUT_DIR_BASE) / mode_name
        current_detailed_log_dir = current_output_dir / DETAILED_LOG_DIR_NAME
        current_plot_dir = current_output_dir / PLOT_DIR_NAME
        current_output_dir.mkdir(parents=True, exist_ok=True)
        current_detailed_log_dir.mkdir(parents=True, exist_ok=True)
        current_plot_dir.mkdir(parents=True, exist_ok=True)

        # --- Initialize Multiprocessing Components ---
        try:
            frame_input_queue = Queue(maxsize=2); results_output_queue = Queue(maxsize=2); stop_event = Event()
        except Exception as e: raise RuntimeError(f"Error initializing multiprocessing components: {e}")

        # Check if engine file exists
        if not Path(YOLOV12_ENGINE_PATH).exists(): raise FileNotFoundError(f"Engine file not found: {YOLOV12_ENGINE_PATH}")

        # --- Start Inference Process ---
        try:
            inference_proc = Process(target=inference_process_worker, args=(YOLOV12_ENGINE_PATH, CONF_THRESHOLD, DEVICE, frame_input_queue, results_output_queue, stop_event), daemon=True)
            inference_proc.start(); print(f"Inference process started (PID: {inference_proc.pid}). Waiting for class names...")
        except Exception as e: raise RuntimeError(f"Error starting inference process: {e}")

        # --- Wait for Initial Message (Class Names or Error) ---
        try:
            message_type, data = results_output_queue.get(timeout=60)
            if message_type == "CLASS_NAMES": class_names = data; print(f"Received class names: {class_names}");
            elif message_type == "ERROR": raise RuntimeError(f"Inference process failed during initialization: {data}")
            else: raise RuntimeError(f"Received unexpected initial message from worker: {message_type}")
        except Empty: raise RuntimeError("Timed out waiting for class names or error from inference process.")
        except Exception as e: raise RuntimeError(f"Error getting initial data from inference process: {e}")

        if not class_names: print("Warning: Proceeding without valid class names. Using default CLASS_NAMES dict."); class_names = CLASS_NAMES

        # --- Initialize Video Capture ---
        try:
            source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE; cap = cv2.VideoCapture(source)
            if not cap.isOpened(): raise ValueError(f"Could not open video source: {VIDEO_SOURCE}")
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS);
            if frame_rate is None or frame_rate <= 0: frame_rate = 30.0; print(f"Warning: Using default FPS: {frame_rate}")
            print(f"Video source opened: {VIDEO_SOURCE} ({original_width}x{original_height} @ {frame_rate:.2f} FPS)")
            latencies = defaultdict(lambda: deque(maxlen=int(frame_rate * 5))) # Update deque maxlen
        except Exception as e: raise RuntimeError(f"Error opening video source: {e}")

        baseline_y = int(original_height * 3 / 4)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        det_log_path = current_detailed_log_dir / f"{mode_name}_{timestamp_str}_detections.csv"
        try:
            det_log_file = open(det_log_path, 'w', newline='', encoding='utf-8'); det_log_writer = csv.writer(det_log_file)
            det_log_writer.writerow(["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]); print(f"Logging detailed detections to: {det_log_path}")
        except Exception as e: print(f"Warning: Error opening detection log file '{det_log_path}': {e}"); det_log_writer = None; det_log_file=None

        # --- Initialize Tracking Variables ---
        frame_count_read = 0; frame_count_processed = 0
        processing_start_time = time.perf_counter()
        
        # Parameter for FPS calculation
        fps_update_interval = 1.0  # 1초마다 FPS 업데이트
        last_fps_update_time = processing_start_time
        frames_since_last_update = 0
        current_fps = 0

        # --- Main Serial IPC Processing Loop ---
        while True:
            loop_run_time = time.perf_counter() # Use this for duration check
            if (loop_run_time - processing_start_time) > RUN_DURATION_SEC:
                print("\nRun duration reached."); break

            # Initialize latencies for this frame
            read_ms, resize_ms, ipc_send_ms = np.nan, np.nan, np.nan
            infer_ms, ipc_recv_ms, draw_ms, end2end_ms = np.nan, np.nan, np.nan, np.nan
            current_frame_id = -1 # Reset frame id for this cycle

            # --- 1. Read Frame ---
            read_start = time.perf_counter(); ret, frame_orig = cap.read(); read_end = time.perf_counter()
            if not ret: print("\nEnd of video source."); break
            read_ms = (read_end - read_start) * 1000; latencies['read'].append(read_ms)
            current_frame_id = frame_count_read; frame_count_read += 1

            # --- Start E2E Timer ---
            e2e_start_time = time.perf_counter()

            # --- 2. Resize Frame ---
            resize_start = time.perf_counter()
            frame_resized = cv2.resize(frame_orig, RESIZE_DIM)
            resize_end = time.perf_counter()
            resize_ms = (resize_end - resize_start) * 1000; latencies['resize'].append(resize_ms)

            # --- 3. Send to Inference Process ---
            ipc_send_start = time.perf_counter()
            try:
                frame_input_queue.put((current_frame_id, frame_resized), timeout=1.0)
                ipc_send_end = time.perf_counter()
                ipc_send_ms = (ipc_send_end - ipc_send_start) * 1000; latencies['ipc_send'].append(ipc_send_ms)
            except Full: print(f"Warning: Input queue full for frame {current_frame_id}. Skipping."); continue
            except Exception as e: print(f"Error putting frame {current_frame_id} to input queue: {e}"); break

            # --- 4. WAIT for and Receive Result ---
            received_correct_result = False; result_data = None
            get_start_time = time.perf_counter(); get_timeout_seconds = 5.0
            while time.perf_counter() - get_start_time < get_timeout_seconds:
                try:
                    result_frame_id, boxes_data_worker, infer_time_ms_worker = results_output_queue.get(timeout=0.1)
                    ipc_recv_time = time.perf_counter()
                    if result_frame_id == current_frame_id:
                        ipc_recv_ms = (ipc_recv_time - get_start_time) * 1000; latencies['ipc_recv'].append(ipc_recv_ms)
                        infer_ms = infer_time_ms_worker; latencies['infer'].append(infer_ms)
                        result_data = boxes_data_worker; received_correct_result = True; break
                    else: print(f"Warning: Received result for frame {result_frame_id} while expecting {current_frame_id}. Discarding.")
                except Empty: continue
                except (BrokenPipeError, EOFError): print("Error: Result queue broken while waiting."); raise
                except Exception as e: print(f"Error getting result from output queue: {e}"); raise
            if not received_correct_result: print(f"Error: Timed out waiting for result for frame {current_frame_id}."); break

            # --- 5. Draw Detections ---
            draw_start = time.perf_counter()
            drawn_frame = draw_detections_yolov12(frame_orig, result_data, class_names, baseline_y)
            draw_end = time.perf_counter()
            draw_ms = (draw_end - draw_start) * 1000; latencies['draw'].append(draw_ms)

            # --- End E2E Timer ---
            e2e_end_time = time.perf_counter()
            end2end_ms = (e2e_end_time - e2e_start_time) * 1000; latencies['end2end'].append(end2end_ms)
            frame_count_processed += 1
            
            # FPS calculation and display
            frames_since_last_update += 1
            current_time = time.perf_counter()
            time_elapsed = current_time - last_fps_update_time
            
            if time_elapsed >= fps_update_interval:
                current_fps = frames_since_last_update / time_elapsed
                print(f"\rFPS: {current_fps:.2f} | Processed frames: {frame_count_processed} | "
                      f"Inference time: {infer_ms:.1f}ms | Total processing time: {end2end_ms:.1f}ms", end="")
                frames_since_last_update = 0
                last_fps_update_time = current_time

            # --- 7. Log Detections to CSV ---
            if det_log_writer: log_detections_yolov12(current_frame_id, result_data, det_log_writer)

    except KeyboardInterrupt: print("\nCtrl+C detected. Stopping benchmark...")
    except Exception as e: print(f"\nError during main processing loop: {type(e).__name__}: {e}")
    finally: # --- Cleanup for Serial IPC ---
        print("\nInitiating cleanup...")
        interrupted_cleanup = False
        # 1. Signal the inference process to stop
        print("Signaling inference process to stop...")
        try:
            if 'stop_event' in locals() and stop_event is not None: stop_event.set()
        except Exception as e: print(f"Error setting stop event: {e}"); interrupted_cleanup = True
        # 2. Wait briefly
        try: time.sleep(0.5)
        except KeyboardInterrupt: print("Cleanup interrupted during initial sleep."); interrupted_cleanup = True
        
        # --- Process Cleanup ---
        print("Waiting for inference process to join...")
        try:
            if 'inference_proc' in locals() and inference_proc is not None and inference_proc.is_alive():
                inference_proc.join(timeout=5.0)
        except KeyboardInterrupt: print("Cleanup interrupted waiting for process join."); interrupted_cleanup = True
        except Exception as e: print(f"Error joining inference process: {e}"); interrupted_cleanup = True
        try:
            if 'inference_proc' in locals() and inference_proc is not None and inference_proc.is_alive():
                print("Terminating..."); inference_proc.terminate(); inference_proc.join(timeout=2.0)
                if inference_proc.is_alive(): print("Warning: Failed to terminate inference process.")
                else: print("Inference process terminated.")
        except KeyboardInterrupt: print("Cleanup interrupted during process termination."); interrupted_cleanup = True
        except Exception as e: print(f"Error terminating inference process: {e}"); interrupted_cleanup = True
        finally:
             if 'inference_proc' in locals() and inference_proc is not None and not inference_proc.is_alive(): inference_proc = None
        # --- Queue Cleanup ---
        print("Cleaning and closing queues...")
        # Input Queue
        try:
            if 'frame_input_queue' in locals() and frame_input_queue is not None:
                frame_input_queue.cancel_join_thread()
                while not frame_input_queue.empty():
                    try:
                        frame_input_queue.get_nowait()
                    except Empty:
                        break # Exit loop when queue is empty
                    except Exception as e_get_inp:
                        break # Exit loop on other errors too
                frame_input_queue.close(); print("Input queue closed.")
        except KeyboardInterrupt: print("Cleanup interrupted closing input queue."); interrupted_cleanup = True
        except Exception as e: print(f"Error closing input queue: {e}")
        # Output Queue
        try:
            if 'results_output_queue' in locals() and results_output_queue is not None:
                results_output_queue.cancel_join_thread()
                while not results_output_queue.empty():
                    try:
                        results_output_queue.get_nowait()
                    except Empty:
                        break # Exit loop when queue is empty
                    except Exception as e_get_out:
                         break # Exit loop on other errors too
                results_output_queue.close(); print("Output queue closed.")
        except KeyboardInterrupt: print("Cleanup interrupted closing output queue."); interrupted_cleanup = True
        except Exception as e: print(f"Error closing output queue: {e}")
        
        # --- Other Resource Cleanup ---
        try:
            if 'cap' in locals() and cap is not None and cap.isOpened(): cap.release(); print("Video capture released.")
        except KeyboardInterrupt: print("Cleanup interrupted releasing video capture."); interrupted_cleanup = True
        except Exception as e: print(f"Error releasing video capture: {e}")
        try:
            if 'det_log_file' in locals() and det_log_file is not None and not det_log_file.closed: det_log_file.close(); print("Detection log file closed.")
        except KeyboardInterrupt: print("Cleanup interrupted closing log file."); interrupted_cleanup = True
        except Exception as e: print(f"Error closing detection log file: {e}")
        print("Resource cleanup finished." + (" (Interrupted)" if interrupted_cleanup else ""))


    # --- Aggregate, Save, Plot (After finally block) ---
    print("\nCalculating final statistics...")
    if 'processing_start_time' not in locals(): processing_start_time = time.perf_counter() # Fallback
    total_time = time.perf_counter() - processing_start_time
    avg_fps = frame_count_processed / total_time if total_time > 0 else 0

    # Calculate average latencies
    avg_latencies = {}
    latency_keys_final = ['read', 'resize', 'ipc_send', 'infer', 'ipc_recv', 'draw', 'end2end']
    for k in latency_keys_final:
        if k in latencies: avg_latencies[f"avg_latency_{k}_ms"] = np.mean(latencies[k]) if latencies[k] else 0.0
        else: avg_latencies[f"avg_latency_{k}_ms"] = 0.0

    summary_results = {
        "mode": mode_name,
        "total_frames_processed": frame_count_processed, 
        "total_time_sec": total_time,
        "avg_fps": avg_fps, 
        **avg_latencies
    }
    
    print(f"\n--- YOLOv12 Benchmark Summary ({mode_name}) ---")
    for key, value in summary_results.items(): 
        print(f"{key}: {value:.2f}" if isinstance(value, (float, np.float64)) and not np.isnan(value) else f"{key}: {value if value is not None else 'N/A'}")

    # Save summary CSV
    summary_csv_path = Path(OUTPUT_DIR_BASE) / SUMMARY_CSV; file_exists = summary_csv_path.is_file()
    try:
        headers = sorted(summary_results.keys())
        with open(summary_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            if not file_exists or os.path.getsize(summary_csv_path) == 0: writer.writeheader()
            row_to_write = {}
            for k in headers:
                v = summary_results.get(k)
                if isinstance(v, (float, np.float64)): row_to_write[k] = f"{v:.4f}" if not np.isnan(v) else ""
                elif v is None: row_to_write[k] = ""
                else: row_to_write[k] = v
            writer.writerow(row_to_write)
        print(f"Appended summary results to: {summary_csv_path}")
    except Exception as e: print(f"Error saving summary CSV: {e}")

    # Generate plots
    plot_results(latencies, current_output_dir, mode_name)

    # GC and Cache clear moved here
    print("Running garbage collection..."); gc.collect()
    if torch.cuda.is_available(): print("Clearing CUDA cache..."); torch.cuda.empty_cache()

    print(f"YOLOv12 Benchmark Function Finished ({mode_name}).")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Set start method because we are using multiprocessing again
    try:
        current_method = get_start_method(allow_none=True)
        if current_method != 'spawn': set_start_method('spawn', force=True); print(f"Set multiprocessing start method to 'spawn'")
        else: print(f"Multiprocessing start method already set to 'spawn'")
    except RuntimeError: print(f"Warning: Multiprocessing context already set. Ignoring.")
    except Exception as e: print(f"Warning: Could not set start method: {e}")

    # Run the benchmark
    print(f"Starting benchmark execution ({OUTPUT_DIR_BASE})...")
    try: 
        run_yolov12_benchmark()
    except Exception as main_exec_err:
         print(f"\n--- An unexpected error occurred during benchmark execution ---")
         print(f"Error Type: {type(main_exec_err).__name__}")
         print(f"Error Details: {main_exec_err}")

    print("Benchmark script finished.") # Final message
