#!/usr/bin/env python3
"""
Custom YOLO Training Script
============================
Train YOLOv8 on your custom ball dataset

USAGE:
    python train_custom_yolo.py --data ball_dataset/data.yaml
    python train_custom_yolo.py --data ball_dataset/data.yaml --epochs 100 --device cpu
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not installed. Run: pip install ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO ball detection model")
    parser.add_argument('-d', '--data', required=True, help="Path to data.yaml file (from Roboflow)")
    parser.add_argument('--model', default='yolov8n.pt', help="Base model (yolov8n.pt, yolov8s.pt, etc)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--device', default='cpu', help="Device (cpu or cuda)")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--project', default='trained_models', help="Project name")
    parser.add_argument('--name', default='custom_ball', help="Run name")
    
    args = parser.parse_args()
    
    # Verify data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {args.data}")
        print("\n📝 Make sure you:")
        print("   1. Downloaded dataset from Roboflow")
        print("   2. Extracted the zip file")
        print("   3. Path to data.yaml is correct")
        return
    
    print("\n" + "="*60)
    print("YOLO CUSTOM TRAINING")
    print("="*60)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch}")
    print("="*60 + "\n")
    
    # Load base model
    print("[INFO] Loading base model...")
    model = YOLO(args.model)
    
    # Train model
    print("[INFO] Starting training...")
    print("[INFO] This may take 1-3 hours depending on your hardware\n")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save=True,
        project=args.project,
        name=args.name,
        plots=True,
        verbose=True
    )
    
    # Get best model path
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {best_model_path.absolute()}")
    print(f"\n📊 Training results:")
    print(f"   - Charts: {Path(args.project) / args.name}")
    print(f"   - Weights: {Path(args.project) / args.name / 'weights'}")
    print(f"\n🚀 NEXT STEP: Use your trained model")
    print(f"   python yolo_ball_tracking.py --model {best_model_path} --hud")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

