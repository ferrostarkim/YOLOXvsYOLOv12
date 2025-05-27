from ultralytics import YOLO
import os
import multiprocessing

# Dataset path
DATASET_PATH = 'C:/Users/kimjn/YOLOv12/dataset/data.yaml'

# Output directory
OUTPUT_DIR = 'yolov12_custom'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    # Load pre-trained YOLOv12 model
    model = YOLO('C:/Users/kimjn/YOLOv12/yolov12s.pt')
    
    # Adjust the output layer to match the number of new classes (automatically done)
    # Start model fine-tuning
    results = model.train(
        data=DATASET_PATH,
        epochs=300,                       # Number of epochs (fine-tuning, so set it small)
        imgsz=640,
        batch=8,
        workers=4,
        device=0,
        project=OUTPUT_DIR,
        name='exp',
        exist_ok=True,
        patience=50,
        save=True,
        optimizer='AdamW',
        lr0=0.001,                        # Learning rate (fine-tuning, so set it small)
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        dropout=0.0,                      # In fine-tuning, reduce or remove dropout
        augment=True,
    )

    print(f"Training complete! Model saved to: {OUTPUT_DIR}/exp/")
    print(f"Final mAP: {results.box.map}")

    # Model evaluation
    metrics = model.val()
    print(f"Validation dataset mAP: {metrics.box.map}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()