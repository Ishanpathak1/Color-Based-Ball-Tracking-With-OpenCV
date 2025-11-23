#!/usr/bin/env python3
"""
Comparison Script: Color-Based vs YOLO-Based Ball Tracking
===========================================================
This script runs both methods side-by-side to compare performance,
accuracy, and robustness in different conditions.

USAGE:
    python compare_methods.py --video test_video.mp4
    python compare_methods.py  # uses webcam
"""

import argparse
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List
import sys

# Check if YOLO is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO not available. Install with: pip install ultralytics")


class ColorBasedDetector:
    """Simple color-based ball detector (traditional method)"""
    
    def __init__(self, lower_hsv=(29, 86, 6), upper_hsv=(64, 255, 255)):
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        self.detections = 0
        
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """Detect balls using color-based method
        
        Returns:
            List of (x, y, radius, confidence) tuples
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.lower, self.upper)
        
        # Clean up mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 3:
                continue
            
            # Use area as confidence proxy
            confidence = min(area / 1000.0, 1.0)
            detections.append((int(x), int(y), radius, confidence))
            self.detections += 1
        
        return detections


class YOLODetector:
    """YOLO-based ball detector (ML method)"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.detections = 0
        
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """Detect balls using YOLO
        
        Returns:
            List of (x, y, radius, confidence) tuples
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                # Check if it's a ball
                if 'ball' in class_name.lower() or class_name == 'sports ball':
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = max(x2 - x1, y2 - y1) / 2
                    
                    detections.append((center_x, center_y, radius, conf))
                    self.detections += 1
        
        return detections


class PerformanceMetrics:
    """Track and compare performance metrics"""
    
    def __init__(self):
        self.color_fps = []
        self.yolo_fps = []
        self.color_detections = 0
        self.yolo_detections = 0
        self.frames_processed = 0
        
    def update(self, color_time: float, yolo_time: float,
               color_dets: int, yolo_dets: int):
        """Update metrics"""
        self.color_fps.append(1.0 / color_time if color_time > 0 else 0)
        self.yolo_fps.append(1.0 / yolo_time if yolo_time > 0 else 0)
        self.color_detections += color_dets
        self.yolo_detections += yolo_dets
        self.frames_processed += 1
        
    def get_summary(self) -> str:
        """Get summary statistics"""
        avg_color_fps = np.mean(self.color_fps) if self.color_fps else 0
        avg_yolo_fps = np.mean(self.yolo_fps) if self.yolo_fps else 0
        
        summary = "\n" + "="*60 + "\n"
        summary += "PERFORMANCE COMPARISON SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"Frames Processed: {self.frames_processed}\n"
        summary += "\n--- COLOR-BASED (Traditional) ---\n"
        summary += f"  Average FPS: {avg_color_fps:.2f}\n"
        summary += f"  Total Detections: {self.color_detections}\n"
        summary += f"  Avg Detections/Frame: {self.color_detections/max(1, self.frames_processed):.2f}\n"
        summary += "\n--- YOLO-BASED (Deep Learning) ---\n"
        summary += f"  Average FPS: {avg_yolo_fps:.2f}\n"
        summary += f"  Total Detections: {self.yolo_detections}\n"
        summary += f"  Avg Detections/Frame: {self.yolo_detections/max(1, self.frames_processed):.2f}\n"
        summary += "\n--- COMPARISON ---\n"
        summary += f"  Speed Ratio (Color/YOLO): {avg_color_fps/max(0.1, avg_yolo_fps):.2f}x\n"
        
        if avg_color_fps > avg_yolo_fps:
            summary += f"  Winner (Speed): COLOR-BASED ({avg_color_fps:.1f} FPS)\n"
        else:
            summary += f"  Winner (Speed): YOLO-BASED ({avg_yolo_fps:.1f} FPS)\n"
        
        summary += "\n--- NOTES ---\n"
        summary += "  - Color-based is faster but sensitive to lighting\n"
        summary += "  - YOLO is more robust but requires more computation\n"
        summary += "  - YOLO works with any ball color/lighting\n"
        summary += "  - Color-based needs HSV tuning per environment\n"
        summary += "="*60 + "\n"
        
        return summary


def draw_detections(frame: np.ndarray, detections: List[Tuple],
                    color: Tuple[int, int, int], label: str):
    """Draw detections on frame"""
    for x, y, radius, conf in detections:
        cv2.circle(frame, (x, y), int(radius), color, 2)
        cv2.circle(frame, (x, y), 3, color, -1)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x - 30, y - int(radius) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Compare Color-based vs YOLO tracking")
    parser.add_argument('-v', '--video', help="Path to video file")
    parser.add_argument('--yolo-model', default='yolov8n.pt', help="YOLO model path")
    parser.add_argument('--lower-hsv', default='29,86,6', help="Lower HSV (H,S,V)")
    parser.add_argument('--upper-hsv', default='64,255,255', help="Upper HSV (H,S,V)")
    parser.add_argument('--split-view', action='store_true', help="Show split screen comparison")
    
    args = parser.parse_args()
    
    # Parse HSV values
    lower_hsv = tuple(map(int, args.lower_hsv.split(',')))
    upper_hsv = tuple(map(int, args.upper_hsv.split(',')))
    
    # Initialize video
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return
    
    # Initialize detectors
    print("[INFO] Initializing Color-based detector...")
    color_detector = ColorBasedDetector(lower_hsv, upper_hsv)
    
    yolo_detector = None
    if YOLO_AVAILABLE:
        print("[INFO] Initializing YOLO detector...")
        try:
            yolo_detector = YOLODetector(args.yolo_model)
        except Exception as e:
            print(f"[WARNING] Could not initialize YOLO: {e}")
            yolo_detector = None
    
    if yolo_detector is None:
        print("[INFO] Running COLOR-BASED method only")
    else:
        print("[INFO] Running BOTH methods for comparison")
    
    # Initialize metrics
    metrics = PerformanceMetrics()
    
    print("[INFO] Starting comparison... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for processing
        frame = cv2.resize(frame, (800, 600))
        
        # Test Color-based method
        t_start = time.time()
        color_dets = color_detector.detect(frame)
        color_time = time.time() - t_start
        
        # Test YOLO method
        yolo_time = 0
        yolo_dets = []
        if yolo_detector:
            t_start = time.time()
            yolo_dets = yolo_detector.detect(frame)
            yolo_time = time.time() - t_start
        
        # Update metrics
        metrics.update(color_time, yolo_time, len(color_dets), len(yolo_dets))
        
        # Visualization
        if args.split_view and yolo_detector:
            # Split screen view
            frame_color = frame.copy()
            frame_yolo = frame.copy()
            
            draw_detections(frame_color, color_dets, (0, 255, 0), "Color")
            draw_detections(frame_yolo, yolo_dets, (0, 0, 255), "YOLO")
            
            # Add labels
            cv2.putText(frame_color, f"COLOR-BASED {1/color_time:.1f}FPS", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_yolo, f"YOLO-BASED {1/yolo_time:.1f}FPS" if yolo_time > 0 else "YOLO-BASED",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Combine
            combined = np.hstack([frame_color, frame_yolo])
            cv2.imshow('Comparison: Color (Left) vs YOLO (Right)', combined)
        else:
            # Single view with both detections
            draw_detections(frame, color_dets, (0, 255, 0), "Color")
            if yolo_detector:
                draw_detections(frame, yolo_dets, (0, 0, 255), "YOLO")
            
            # Add info
            info_text = [
                f"Color: {len(color_dets)} dets, {1/color_time:.1f} FPS",
            ]
            if yolo_detector:
                info_text.append(f"YOLO: {len(yolo_dets)} dets, {1/yolo_time:.1f} FPS" if yolo_time > 0 else "YOLO: Processing...")
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 25
            
            cv2.imshow('Method Comparison', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(metrics.get_summary())


if __name__ == '__main__':
    main()

