#!/usr/bin/env python3
"""
Smart Semi-Automatic Labeling
==============================
Goes through ALL images:
  - If YOLO detects ball → Shows it, you confirm or adjust
  - If YOLO misses → You draw manually
  
Controls:
  SPACE/'s' - Accept YOLO detection and save
  Click & Drag - Draw/add new box
  'd' - Delete last box
  'r' - Reset all boxes
  'n' - Skip (no ball)
  'q' - Quit
"""

import cv2
import argparse
from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Install ultralytics: pip install ultralytics")
    exit(1)


class SmartAnnotator:
    def __init__(self, model):
        self.model = model
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes = []  # List of (x1, y1, x2, y2)
        self.current_img = None
        self.display_img = None
        self.yolo_suggested = False
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.boxes.append((x1, y1, x2, y2))
                self.start_point = None
                self.end_point = None
                self.yolo_suggested = False  # User added manually
    
    def get_yolo_suggestions(self, img, conf_threshold=0.25):
        """Get YOLO ball detections"""
        results = self.model(img, conf=conf_threshold, verbose=False)
        
        suggestions = []
        BALL_CLASSES = {'sports ball', 'baseball', 'frisbee'}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                if class_name in BALL_CLASSES or 'ball' in class_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    suggestions.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return suggestions
    
    def load_image(self, img):
        """Load new image and get YOLO suggestions"""
        self.current_img = img
        self.boxes = []
        self.yolo_suggested = False
        
        # Get YOLO suggestions
        suggestions = self.get_yolo_suggestions(img)
        
        if suggestions:
            # Use YOLO detections as starting point
            self.boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf in suggestions]
            self.yolo_suggested = True
            return len(suggestions)
        return 0
    
    def draw_boxes(self):
        self.display_img = self.current_img.copy()
        h, w = self.display_img.shape[:2]
        
        # Draw saved boxes
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            color = (0, 255, 255) if self.yolo_suggested else (0, 255, 0)
            cv2.rectangle(self.display_img, (x1, y1), (x2, y2), color, 2)
            label = f"YOLO {i+1}" if self.yolo_suggested else f"Ball {i+1}"
            cv2.putText(self.display_img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current box being drawn
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(self.display_img, self.start_point, self.end_point, (255, 0, 255), 2)
        
        # Status and instructions
        status = "YOLO AUTO-DETECTED" if self.yolo_suggested else "MANUAL MODE"
        color = (0, 255, 255) if self.yolo_suggested else (255, 255, 255)
        
        cv2.putText(self.display_img, f"{status} | Boxes: {len(self.boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        instructions = "SPACE=Accept | Draw=Add box | d=Delete | r=Reset | n=Skip | q=Quit"
        cv2.putText(self.display_img, instructions,
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def save_yolo_format(self, save_path, img_width, img_height):
        """Save boxes in YOLO format"""
        with open(save_path, 'w') as f:
            for x1, y1, x2, y2 in self.boxes:
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Smart semi-automatic labeling")
    parser.add_argument('--images', default='training_data', help="Images directory")
    parser.add_argument('--output', default='labeled_data', help="Output directory")
    parser.add_argument('--model', default='yolov8n.pt', help="YOLO model")
    parser.add_argument('--conf', type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument('--start', type=int, default=0, help="Start from image N")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_images = output_dir / 'images'
    output_labels = output_dir / 'labels'
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    
    if not image_files:
        print("[ERROR] No images found!")
        return
    
    print("\n" + "="*60)
    print("SMART SEMI-AUTOMATIC LABELING")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"YOLO Model: {args.model}")
    print("\nHow it works:")
    print("  - YOLO detects balls automatically (yellow boxes)")
    print("  - You can accept (SPACE), adjust (draw new), or skip")
    print("  - If YOLO misses, you draw manually")
    print("\nControls:")
    print("  SPACE/'s'   - Accept and save")
    print("  Click&Drag  - Draw/add new box")
    print("  'd'         - Delete last box")
    print("  'r'         - Reset all boxes")
    print("  'n'         - Skip (no ball)")
    print("  'q'         - Quit")
    print("="*60 + "\n")
    
    # Load YOLO
    print("[INFO] Loading YOLO model...")
    model = YOLO(args.model)
    
    annotator = SmartAnnotator(model)
    cv2.namedWindow('Smart Labeling')
    cv2.setMouseCallback('Smart Labeling', annotator.mouse_callback)
    
    idx = args.start
    saved_count = 0
    skipped_count = 0
    yolo_auto_count = 0
    manual_count = 0
    
    while idx < len(image_files):
        img_path = image_files[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue
        
        h, w = img.shape[:2]
        
        # Load image and get YOLO suggestions
        num_suggestions = annotator.load_image(img)
        
        print(f"\n[{idx+1}/{len(image_files)}] {img_path.name}")
        if num_suggestions > 0:
            print(f"  ✅ YOLO found {num_suggestions} ball(s) - Review and press SPACE to accept")
        else:
            print(f"  ⚠️  YOLO missed - Draw box manually or press 'n' to skip")
        
        while True:
            annotator.draw_boxes()
            cv2.imshow('Smart Labeling', annotator.display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                print(f"\n✅ Progress: {saved_count} labeled, {skipped_count} skipped")
                print(f"   Auto: {yolo_auto_count}, Manual: {manual_count}")
                return
            
            elif key == ord(' ') or key == ord('s'):  # Save
                if annotator.boxes:
                    # Save image
                    import shutil
                    dest_img = output_images / img_path.name
                    shutil.copy2(str(img_path), str(dest_img))
                    
                    # Save label
                    label_path = output_labels / f"{img_path.stem}.txt"
                    annotator.save_yolo_format(label_path, w, h)
                    
                    saved_count += 1
                    if annotator.yolo_suggested:
                        yolo_auto_count += 1
                        print(f"  ✅ Saved YOLO detection")
                    else:
                        manual_count += 1
                        print(f"  ✅ Saved manual label")
                    break
                else:
                    print("  ⚠️  No boxes! Draw one or press 'n' to skip")
            
            elif key == ord('n'):  # Skip
                skipped_count += 1
                print(f"  ⏭️  Skipped")
                break
            
            elif key == ord('d'):  # Delete last box
                if annotator.boxes:
                    annotator.boxes.pop()
                    annotator.yolo_suggested = False
                    print(f"  🗑️  Deleted box, {len(annotator.boxes)} remaining")
            
            elif key == ord('r'):  # Reset
                annotator.boxes = []
                annotator.yolo_suggested = False
                print(f"  🔄 Reset all boxes")
        
        idx += 1
        
        # Progress update every 50 images
        if idx % 50 == 0:
            print(f"\n📊 Progress: {idx}/{len(image_files)} | Saved: {saved_count} | Auto: {yolo_auto_count} | Manual: {manual_count}")
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ LABELING COMPLETE!")
    print("="*60)
    print(f"  Total processed: {idx}")
    print(f"  Saved: {saved_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  YOLO auto: {yolo_auto_count}")
    print(f"  Manual: {manual_count}")
    print(f"  Final dataset: {len(list(output_labels.glob('*.txt')))} images")
    print("\n📝 NEXT: Train your model")
    print(f"  python train_custom_yolo.py --data {output_dir / 'data.yaml'} --epochs 50")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

