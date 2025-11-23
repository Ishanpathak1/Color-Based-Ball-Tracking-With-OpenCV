#!/usr/bin/env python3
"""
Manual Labeling for Missed Images
==================================
Label images that YOLO missed - simple click & drag to draw boxes

Controls:
  Click & Drag - Draw bounding box around ball
  'r' - Reset current box
  's' - Save and go to next
  'n' - Skip this image (no ball)
  'q' - Quit
"""

import cv2
import argparse
from pathlib import Path
import shutil

class SimpleAnnotator:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes = []
        self.current_img = None
        self.display_img = None
        
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
            
            # Save box
            if self.start_point and self.end_point:
                self.boxes.append((self.start_point, self.end_point))
                self.start_point = None
                self.end_point = None
    
    def draw_boxes(self):
        self.display_img = self.current_img.copy()
        
        # Draw saved boxes
        for i, (pt1, pt2) in enumerate(self.boxes):
            cv2.rectangle(self.display_img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(self.display_img, f"Ball {i+1}", (pt1[0], pt1[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw current box being drawn
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(self.display_img, self.start_point, self.end_point, (255, 255, 0), 2)
        
        # Instructions
        h, w = self.display_img.shape[:2]
        cv2.putText(self.display_img, f"Boxes: {len(self.boxes)} | Draw box around ball", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_img, "s=Save & Next | r=Reset | n=Skip | q=Quit",
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def save_yolo_format(self, save_path, img_width, img_height):
        """Save boxes in YOLO format"""
        with open(save_path, 'w') as f:
            for pt1, pt2 in self.boxes:
                x1, y1 = pt1
                x2, y2 = pt2
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Class 0 for ball
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Manual labeling for missed images")
    parser.add_argument('--original', default='training_data', help="Original images directory")
    parser.add_argument('--labeled', default='labeled_data', help="Labeled data directory")
    parser.add_argument('--start', type=int, default=0, help="Start from image N")
    
    args = parser.parse_args()
    
    original_dir = Path(args.original)
    labeled_dir = Path(args.labeled)
    images_dir = labeled_dir / 'images'
    labels_dir = labeled_dir / 'labels'
    
    # Get all original images
    all_images = sorted(list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.png')))
    
    # Get already labeled images
    labeled_images = set(p.stem for p in images_dir.glob('*'))
    
    # Find missed images
    missed_images = [img for img in all_images if img.stem not in labeled_images]
    
    if not missed_images:
        print("[INFO] No missed images! All images are labeled.")
        return
    
    print("\n" + "="*60)
    print("MANUAL LABELING MODE - Missed Images")
    print("="*60)
    print(f"Total original images: {len(all_images)}")
    print(f"Already labeled: {len(labeled_images)}")
    print(f"Missed images to label: {len(missed_images)}")
    print("\nControls:")
    print("  Click & Drag - Draw box around ball")
    print("  'r' - Reset all boxes")
    print("  's' - Save and go to next image")
    print("  'n' - Skip (no ball in this image)")
    print("  'q' - Quit")
    print("="*60 + "\n")
    
    annotator = SimpleAnnotator()
    cv2.namedWindow('Manual Labeling')
    cv2.setMouseCallback('Manual Labeling', annotator.mouse_callback)
    
    idx = args.start
    saved_count = 0
    skipped_count = 0
    
    while idx < len(missed_images):
        img_path = missed_images[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue
        
        h, w = img.shape[:2]
        annotator.current_img = img
        annotator.boxes = []
        
        print(f"\n[{idx+1}/{len(missed_images)}] Labeling: {img_path.name}")
        
        while True:
            annotator.draw_boxes()
            cv2.imshow('Manual Labeling', annotator.display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                print(f"\n✅ Labeled {saved_count} images, skipped {skipped_count}")
                return
            
            elif key == ord('s'):  # Save
                if annotator.boxes:
                    # Copy image to labeled directory
                    dest_img = images_dir / img_path.name
                    shutil.copy2(str(img_path), str(dest_img))
                    
                    # Save label
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    annotator.save_yolo_format(label_path, w, h)
                    
                    saved_count += 1
                    print(f"✅ Saved with {len(annotator.boxes)} box(es)")
                    break
                else:
                    print("⚠️  No boxes drawn! Draw a box or press 'n' to skip")
            
            elif key == ord('n'):  # Skip
                skipped_count += 1
                print(f"⏭️  Skipped (no ball)")
                break
            
            elif key == ord('r'):  # Reset
                annotator.boxes = []
                print("🔄 Reset boxes")
        
        idx += 1
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ MANUAL LABELING COMPLETE!")
    print("="*60)
    print(f"  Images processed: {idx}")
    print(f"  New labels created: {saved_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total labeled now: {len(list(labels_dir.glob('*.txt')))}")
    print("\n📝 NEXT: Train your model")
    print(f"  python train_custom_yolo.py --data {labeled_dir / 'data.yaml'} --epochs 50")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

