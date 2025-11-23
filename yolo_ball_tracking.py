#!/usr/bin/env python3
"""
YOLO-Based Ball Tracking with Deep Learning
============================================
A modern ball tracking system using YOLOv8 for detection combined with
advanced tracking algorithms for trajectory prediction and analysis.

USAGE:
    python yolo_ball_tracking.py --video ball_video.mp4
    python yolo_ball_tracking.py  # uses webcam
    python yolo_ball_tracking.py --video test.mp4 --model yolov8n.pt --display-metrics --hud

Features:
    - YOLOv8 deep learning detection (works in any lighting)
    - Multi-ball tracking with unique IDs
    - Trajectory prediction using physics
    - Performance metrics (FPS, confidence scores)
    - Data logging for analysis
    - Real-time visualization with HUD
"""

import argparse
import time
import cv2
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import json
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics YOLO not installed. Install with: pip install ultralytics")


@dataclass
class TrackedBall:
    """Represents a tracked ball with its history and properties"""
    track_id: int
    class_name: str
    color: Tuple[int, int, int]
    buffer_len: int = 64
    
    # History tracking
    positions: deque = field(default_factory=lambda: deque(maxlen=64))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=64))
    confidences: deque = field(default_factory=lambda: deque(maxlen=64))
    
    # Current state
    last_center: Optional[Tuple[int, int]] = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_confidence: float = 0.0
    velocity: Optional[np.ndarray] = None
    speed: float = 0.0
    predicted_position: Optional[Tuple[int, int]] = None
    frames_missing: int = 0
    
    def __post_init__(self):
        """Initialize deques with proper maxlen"""
        if not isinstance(self.positions, deque):
            self.positions = deque(maxlen=self.buffer_len)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(maxlen=self.buffer_len)
        if not isinstance(self.confidences, deque):
            self.confidences = deque(maxlen=self.buffer_len)
    
    def update(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int], 
               confidence: float, timestamp: float):
        """Update ball state with new detection"""
        self.positions.appendleft(center)
        self.timestamps.appendleft(timestamp)
        self.confidences.appendleft(confidence)
        
        self.last_center = center
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.frames_missing = 0
        
        # Calculate velocity
        self._calculate_velocity()
        self._predict_next_position()
    
    def _calculate_velocity(self):
        """Calculate velocity from recent positions"""
        if len(self.positions) < 2:
            return
        
        for i in range(1, min(5, len(self.positions))):
            if self.timestamps[0] and self.timestamps[i]:
                dt = self.timestamps[0] - self.timestamps[i]
                if dt > 0:
                    dx = self.positions[0][0] - self.positions[i][0]
                    dy = self.positions[0][1] - self.positions[i][1]
                    self.velocity = np.array([dx/dt, dy/dt])
                    self.speed = float(np.linalg.norm(self.velocity))
                    break
    
    def _predict_next_position(self, time_ahead: float = 0.1):
        """Predict future position based on velocity"""
        if self.velocity is not None and self.last_center is not None:
            pred = np.array(self.last_center) + self.velocity * time_ahead
            self.predicted_position = (int(pred[0]), int(pred[1]))
        else:
            self.predicted_position = None
    
    def mark_missing(self):
        """Mark that this ball wasn't detected in current frame"""
        self.frames_missing += 1
    
    def is_active(self, max_missing: int = 30) -> bool:
        """Check if ball should still be tracked"""
        return self.frames_missing < max_missing
    
    def get_avg_confidence(self) -> float:
        """Get average confidence over recent detections"""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class YOLOBallTracker:
    """Main tracker class using YOLO for detection"""
    
    # Ball-related classes from COCO dataset
    BALL_CLASSES = {
        'sports ball': 32,  # Generic sports ball
        'baseball': 36,
        'tennis racket': 38,  # Often detected with tennis balls
        'frisbee': 29,
    }
    
    # Colors for visualization (BGR)
    COLORS = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 0),      # Green
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Light Blue
    ]
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.45, max_distance: float = 100.0):
        """Initialize YOLO tracker"""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed. Run: pip install ultralytics")
        
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        
        # Tracking state
        self.tracked_balls: Dict[int, TrackedBall] = {}
        self.next_id = 0
        self.frame_count = 0
        
        # Statistics
        self.detection_count = 0
        self.total_detections = 0
        
    def detect_balls(self, frame: np.ndarray) -> List[Tuple]:
        """Detect balls in frame using YOLO
        
        Returns:
            List of (center, bbox, confidence, class_name) tuples
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box data
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Check if it's a ball-related class
                class_name = self.model.names[cls_id]
                if class_name in self.BALL_CLASSES or 'ball' in class_name.lower():
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    detections.append((center, bbox, conf, class_name))
                    self.total_detections += 1
        
        return detections
    
    def _match_detection_to_track(self, center: Tuple[int, int], 
                                   tracked_balls: Dict[int, TrackedBall]) -> Optional[int]:
        """Match a detection to existing track using distance"""
        best_match = None
        min_distance = self.max_distance
        
        for track_id, ball in tracked_balls.items():
            if ball.last_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(ball.last_center))
                if dist < min_distance:
                    min_distance = dist
                    best_match = track_id
        
        return best_match
    
    def update(self, frame: np.ndarray, timestamp: float) -> Dict[int, TrackedBall]:
        """Update tracker with new frame
        
        Returns:
            Dictionary of active tracked balls
        """
        self.frame_count += 1
        
        # Detect balls in current frame
        detections = self.detect_balls(frame)
        self.detection_count = len(detections)
        
        # Mark all existing tracks as potentially missing
        for ball in self.tracked_balls.values():
            ball.mark_missing()
        
        # Match detections to existing tracks or create new ones
        matched_tracks = set()
        
        for center, bbox, conf, class_name in detections:
            # Try to match to existing track
            track_id = self._match_detection_to_track(center, self.tracked_balls)
            
            if track_id is not None and track_id not in matched_tracks:
                # Update existing track
                self.tracked_balls[track_id].update(center, bbox, conf, timestamp)
                matched_tracks.add(track_id)
            else:
                # Create new track
                color = self.COLORS[self.next_id % len(self.COLORS)]
                new_ball = TrackedBall(
                    track_id=self.next_id,
                    class_name=class_name,
                    color=color
                )
                new_ball.update(center, bbox, conf, timestamp)
                self.tracked_balls[self.next_id] = new_ball
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # Remove inactive tracks
        self.tracked_balls = {
            tid: ball for tid, ball in self.tracked_balls.items() 
            if ball.is_active()
        }
        
        return self.tracked_balls
    
    def draw_tracks(self, frame: np.ndarray, show_trails: bool = True,
                   show_velocity: bool = True, show_prediction: bool = True):
        """Draw all tracked balls on frame"""
        for track_id, ball in self.tracked_balls.items():
            if ball.last_center is None:
                continue
            
            # Draw bounding box
            if ball.last_bbox is not None:
                x1, y1, x2, y2 = ball.last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), ball.color, 2)
                
                # Draw label
                label = f"ID:{track_id} {ball.class_name} {ball.last_confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), ball.color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw center point
            cv2.circle(frame, ball.last_center, 5, ball.color, -1)
            
            # Draw trail
            if show_trails and len(ball.positions) > 1:
                for i in range(1, len(ball.positions)):
                    if ball.positions[i] is None:
                        continue
                    thickness = max(1, int(np.sqrt(ball.buffer_len / float(i + 1)) * 2.5))
                    alpha = max(0.3, 1.0 - (i / len(ball.positions)))
                    color = tuple(int(c * alpha) for c in ball.color)
                    cv2.line(frame, ball.positions[i-1], ball.positions[i], color, thickness)
            
            # Draw velocity arrow
            if show_velocity and ball.velocity is not None:
                arrow_end = (
                    int(ball.last_center[0] + ball.velocity[0] * 0.1),
                    int(ball.last_center[1] + ball.velocity[1] * 0.1)
                )
                cv2.arrowedLine(frame, ball.last_center, arrow_end, ball.color, 2, tipLength=0.3)
            
            # Draw predicted position
            if show_prediction and ball.predicted_position is not None:
                x_pred, y_pred = ball.predicted_position
                if 0 <= x_pred < frame.shape[1] and 0 <= y_pred < frame.shape[0]:
                    cv2.circle(frame, ball.predicted_position, 8, (255, 255, 0), 2)
                    cv2.line(frame, ball.last_center, ball.predicted_position, 
                            (255, 255, 0), 1, cv2.LINE_AA)
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.tracked_balls),
            'total_detections': self.total_detections,
            'current_detections': self.detection_count,
            'next_id': self.next_id
        }


class DataLogger:
    """Log tracking data for analysis"""
    
    def __init__(self, output_dir: str = 'tracking_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.output_dir / f'tracking_log_{timestamp}.json'
        self.frame_data = []
        
    def log_frame(self, frame_num: int, timestamp: float, 
                  tracked_balls: Dict[int, TrackedBall], fps: float):
        """Log data for current frame"""
        frame_info = {
            'frame': frame_num,
            'timestamp': timestamp,
            'fps': fps,
            'balls': []
        }
        
        for track_id, ball in tracked_balls.items():
            if ball.last_center is not None:
                ball_info = {
                    'id': track_id,
                    'class': ball.class_name,
                    'position': ball.last_center,
                    'bbox': ball.last_bbox,
                    'confidence': ball.last_confidence,
                    'speed': ball.speed,
                    'velocity': ball.velocity.tolist() if ball.velocity is not None else None,
                }
                frame_info['balls'].append(ball_info)
        
        self.frame_data.append(frame_info)
    
    def save(self):
        """Save logged data to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.frame_data, f, indent=2)
        print(f"[INFO] Tracking data saved to: {self.log_file}")


def draw_hud(frame: np.ndarray, tracker: YOLOBallTracker, fps: float):
    """Draw HUD overlay with statistics"""
    stats = tracker.get_statistics()
    
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Active Balls: {stats['active_tracks']}",
        f"Frame: {stats['frame_count']}",
        f"Total Detections: {stats['total_detections']}",
    ]
    
    # Draw HUD background
    hud_height = len(hud_lines) * 25 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw HUD text
    y_offset = 30
    for line in hud_lines:
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 25
    
    # Draw ball-specific info
    y_offset = hud_height + 30
    for track_id, ball in tracker.tracked_balls.items():
        if ball.last_center is not None:
            info = f"Ball {track_id}: Speed={ball.speed:.1f}px/s Conf={ball.last_confidence:.2f}"
            cv2.putText(frame, info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, ball.color, 2, cv2.LINE_AA)
            y_offset += 20


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO-based Ball Tracking")
    parser.add_argument('-v', '--video', help="Path to video file (optional, uses webcam if not provided)")
    parser.add_argument('-m', '--model', default='yolov8n.pt', help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument('--conf', type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument('--iou', type=float, default=0.45, help="IOU threshold (default: 0.45)")
    parser.add_argument('--max-distance', type=float, default=100.0, 
                       help="Max distance for track matching (default: 100.0)")
    parser.add_argument('--buffer', type=int, default=64, help="Trail buffer size (default: 64)")
    parser.add_argument('--hud', action='store_true', help="Display HUD with statistics")
    parser.add_argument('--no-trails', action='store_true', help="Disable trajectory trails")
    parser.add_argument('--no-velocity', action='store_true', help="Disable velocity arrows")
    parser.add_argument('--no-prediction', action='store_true', help="Disable position prediction")
    parser.add_argument('--log-data', action='store_true', help="Log tracking data to file")
    parser.add_argument('--output-dir', default='tracking_data', help="Output directory for logs")
    parser.add_argument('--display-fps', action='store_true', help="Print FPS to console")
    parser.add_argument('--headless', action='store_true', help="Run without display (for testing)")
    
    args = parser.parse_args()
    
    # Check YOLO availability
    if not YOLO_AVAILABLE:
        print("[ERROR] Ultralytics YOLO not installed.")
        print("[INFO] Install with: pip install ultralytics")
        return
    
    # Initialize video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {args.video}")
            return
        print(f"[INFO] Using video file: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return
        print("[INFO] Using webcam")
    
    # Initialize tracker
    tracker = YOLOBallTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        max_distance=args.max_distance
    )
    
    # Initialize data logger
    logger = DataLogger(args.output_dir) if args.log_data else None
    
    # FPS calculation
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0
    
    print("[INFO] Starting tracking... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track balls
        frame_time = time.time()
        tracked_balls = tracker.update(frame, frame_time)
        
        # Draw tracking visualization
        tracker.draw_tracks(
            frame,
            show_trails=not args.no_trails,
            show_velocity=not args.no_velocity,
            show_prediction=not args.no_prediction
        )
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            if args.display_fps:
                print(f"[INFO] FPS: {fps:.2f}")
        
        # Draw HUD
        if args.hud:
            draw_hud(frame, tracker, fps)
        
        # Log data
        if logger:
            logger.log_frame(tracker.frame_count, frame_time, tracked_balls, fps)
        
        # Display
        if not args.headless:
            cv2.imshow('YOLO Ball Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if logger:
        logger.save()
    
    # Print final statistics
    stats = tracker.get_statistics()
    print("\n[INFO] Tracking Complete!")
    print(f"  Total Frames: {stats['frame_count']}")
    print(f"  Total Detections: {stats['total_detections']}")
    print(f"  Unique Balls Tracked: {stats['next_id']}")
    print(f"  Final FPS: {fps:.2f}")


if __name__ == '__main__':
    main()

