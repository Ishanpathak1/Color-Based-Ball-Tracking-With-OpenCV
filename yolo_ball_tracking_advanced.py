#!/usr/bin/env python3
"""
Advanced YOLO Ball Tracking - Optimized for Fast-Moving Objects
================================================================
Enhanced version with Kalman filtering, motion prediction, and 
temporal consistency for tracking fast-moving balls.

USAGE:
    python yolo_ball_tracking_advanced.py --video ball_video.mp4
    python yolo_ball_tracking_advanced.py  # uses webcam
    
Features over basic version:
    - Kalman filter for smooth prediction during occlusions
    - Multi-frame temporal consistency
    - Motion blur preprocessing
    - Adaptive confidence thresholds
    - Track persistence during missed detections
    - Better handling of fast motion
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


class KalmanTracker:
    """Kalman filter for smooth ball tracking and prediction"""
    
    def __init__(self):
        """Initialize Kalman filter for 2D position + velocity tracking"""
        # State: [x, y, vx, vy] - position and velocity
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        
        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], dtype=np.float32)
        
        # Process noise (model uncertainty)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise (sensor uncertainty)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
    
    def initialize(self, position: Tuple[int, int]):
        """Initialize filter with first position"""
        self.kf.statePost = np.array([[position[0]], [position[1]], [0], [0]], dtype=np.float32)
        self.initialized = True
    
    def predict(self) -> Tuple[int, int]:
        """Predict next position"""
        if not self.initialized:
            return (0, 0)
        
        prediction = self.kf.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def update(self, measurement: Tuple[int, int]) -> Tuple[int, int]:
        """Update filter with new measurement"""
        if not self.initialized:
            self.initialize(measurement)
            return measurement
        
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kf.correct(measurement_array)
        
        # Return corrected position
        state = self.kf.statePost
        return (int(state[0]), int(state[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        if not self.initialized:
            return (0.0, 0.0)
        state = self.kf.statePost
        return (float(state[2]), float(state[3]))


@dataclass
class AdvancedTrackedBall:
    """Advanced tracked ball with Kalman filtering"""
    track_id: int
    class_name: str
    color: Tuple[int, int, int]
    buffer_len: int = 64
    
    # Kalman filter
    kalman: KalmanTracker = field(default_factory=KalmanTracker)
    
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
    
    # Detection state
    detection_count: int = 0
    last_detection_frame: int = 0
    
    def __post_init__(self):
        """Initialize deques with proper maxlen"""
        if not isinstance(self.positions, deque):
            self.positions = deque(maxlen=self.buffer_len)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(maxlen=self.buffer_len)
        if not isinstance(self.confidences, deque):
            self.confidences = deque(maxlen=self.buffer_len)
    
    def update_with_detection(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int], 
                             confidence: float, timestamp: float, frame_num: int):
        """Update with actual detection"""
        # Update Kalman filter
        corrected_pos = self.kalman.update(center)
        
        # Store history
        self.positions.appendleft(corrected_pos)
        self.timestamps.appendleft(timestamp)
        self.confidences.appendleft(confidence)
        
        self.last_center = corrected_pos
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.frames_missing = 0
        self.detection_count += 1
        self.last_detection_frame = frame_num
        
        # Get velocity from Kalman filter
        vx, vy = self.kalman.get_velocity()
        self.velocity = np.array([vx, vy])
        self.speed = float(np.linalg.norm(self.velocity))
        
        # Predict next position
        self._predict_next_position()
    
    def update_without_detection(self, timestamp: float):
        """Update with prediction only (no detection)"""
        # Use Kalman prediction
        predicted_pos = self.kalman.predict()
        
        # Store predicted position
        self.positions.appendleft(predicted_pos)
        self.timestamps.appendleft(timestamp)
        self.confidences.appendleft(0.0)  # No confidence for predictions
        
        self.last_center = predicted_pos
        self.frames_missing += 1
        
        # Update velocity and prediction
        vx, vy = self.kalman.get_velocity()
        self.velocity = np.array([vx, vy])
        self.speed = float(np.linalg.norm(self.velocity))
        self._predict_next_position()
    
    def _predict_next_position(self, time_ahead: float = 0.05):
        """Predict future position"""
        if self.velocity is not None and self.last_center is not None:
            pred = np.array(self.last_center) + self.velocity * time_ahead / 0.033  # Assume ~30fps
            self.predicted_position = (int(pred[0]), int(pred[1]))
        else:
            self.predicted_position = None
    
    def is_active(self, max_missing: int = 60) -> bool:
        """Check if track should continue"""
        return self.frames_missing < max_missing
    
    def get_avg_confidence(self) -> float:
        """Get average confidence over recent detections"""
        valid_confs = [c for c in self.confidences if c > 0]
        if not valid_confs:
            return 0.0
        return sum(valid_confs) / len(valid_confs)


class AdvancedYOLOBallTracker:
    """Advanced YOLO tracker with Kalman filtering and motion handling"""
    
    BALL_CLASSES = {
        'sports ball': 32,
        'baseball': 36,
        'tennis racket': 38,
        'frisbee': 29,
    }
    
    COLORS = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 0),      # Green
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Light Blue
    ]
    
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 confidence_threshold: float = 0.25,  # Lower for fast objects
                 iou_threshold: float = 0.45, 
                 max_distance: float = 150.0,  # Larger for fast motion
                 reduce_blur: bool = True):
        """Initialize advanced tracker"""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed")
        
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.reduce_blur = reduce_blur
        
        # Tracking state
        self.tracked_balls: Dict[int, AdvancedTrackedBall] = {}
        self.next_id = 0
        self.frame_count = 0
        
        # Statistics
        self.detection_count = 0
        self.total_detections = 0
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame to handle motion blur"""
        if not self.reduce_blur:
            return frame
        
        # Sharpen the image to reduce motion blur effect
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original to avoid over-sharpening
        return cv2.addWeighted(frame, 0.7, sharpened, 0.3, 0)
    
    def detect_balls(self, frame: np.ndarray) -> List[Tuple]:
        """Detect balls with preprocessing"""
        processed_frame = self.preprocess_frame(frame)
        
        results = self.model(processed_frame, conf=self.conf_threshold, 
                           iou=self.iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                class_name = self.model.names[cls_id]
                if class_name in self.BALL_CLASSES or 'ball' in class_name.lower():
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    detections.append((center, bbox, conf, class_name))
                    self.total_detections += 1
        
        return detections
    
    def _match_detection_to_track(self, center: Tuple[int, int], 
                                   tracked_balls: Dict[int, AdvancedTrackedBall]) -> Optional[int]:
        """Match detection to track using predicted position"""
        best_match = None
        min_distance = self.max_distance
        
        for track_id, ball in tracked_balls.items():
            # Try matching with predicted position first (for fast motion)
            if ball.predicted_position is not None:
                dist = np.linalg.norm(np.array(center) - np.array(ball.predicted_position))
                if dist < min_distance:
                    min_distance = dist
                    best_match = track_id
                    continue
            
            # Fall back to last known position
            if ball.last_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(ball.last_center))
                if dist < min_distance:
                    min_distance = dist
                    best_match = track_id
        
        return best_match
    
    def update(self, frame: np.ndarray, timestamp: float) -> Dict[int, AdvancedTrackedBall]:
        """Update tracker with new frame"""
        self.frame_count += 1
        
        # Detect balls
        detections = self.detect_balls(frame)
        self.detection_count = len(detections)
        
        # Match detections to existing tracks
        matched_tracks = set()
        
        for center, bbox, conf, class_name in detections:
            track_id = self._match_detection_to_track(center, self.tracked_balls)
            
            if track_id is not None and track_id not in matched_tracks:
                # Update existing track
                self.tracked_balls[track_id].update_with_detection(
                    center, bbox, conf, timestamp, self.frame_count
                )
                matched_tracks.add(track_id)
            else:
                # Create new track
                color = self.COLORS[self.next_id % len(self.COLORS)]
                new_ball = AdvancedTrackedBall(
                    track_id=self.next_id,
                    class_name=class_name,
                    color=color
                )
                new_ball.update_with_detection(center, bbox, conf, timestamp, self.frame_count)
                self.tracked_balls[self.next_id] = new_ball
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # Update unmatched tracks with predictions
        for track_id, ball in list(self.tracked_balls.items()):
            if track_id not in matched_tracks:
                ball.update_without_detection(timestamp)
        
        # Remove inactive tracks
        self.tracked_balls = {
            tid: ball for tid, ball in self.tracked_balls.items() 
            if ball.is_active(max_missing=60)  # Keep tracks alive longer
        }
        
        return self.tracked_balls
    
    def draw_tracks(self, frame: np.ndarray, show_trails: bool = True,
                   show_velocity: bool = True, show_prediction: bool = True,
                   show_kalman_state: bool = True):
        """Draw all tracked balls"""
        for track_id, ball in self.tracked_balls.items():
            if ball.last_center is None:
                continue
            
            # Determine if this is a prediction or detection
            is_predicted = ball.frames_missing > 0
            alpha = 0.5 if is_predicted else 1.0
            
            # Draw bounding box (only if recently detected)
            if ball.last_bbox is not None and ball.frames_missing < 3:
                x1, y1, x2, y2 = ball.last_bbox
                box_color = tuple(int(c * alpha) for c in ball.color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw label
                status = "PRED" if is_predicted else f"{ball.last_confidence:.2f}"
                label = f"ID:{track_id} {ball.class_name} {status}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), box_color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw center point
            center_color = tuple(int(c * alpha) for c in ball.color)
            cv2.circle(frame, ball.last_center, 5, center_color, -1)
            
            # Draw trail
            if show_trails and len(ball.positions) > 1:
                for i in range(1, len(ball.positions)):
                    if ball.positions[i] is None:
                        continue
                    thickness = max(1, int(np.sqrt(ball.buffer_len / float(i + 1)) * 2.5))
                    trail_alpha = max(0.2, 1.0 - (i / len(ball.positions)))
                    color = tuple(int(c * trail_alpha) for c in ball.color)
                    cv2.line(frame, ball.positions[i-1], ball.positions[i], color, thickness)
            
            # Draw velocity arrow (longer for fast objects)
            if show_velocity and ball.velocity is not None and ball.speed > 1.0:
                scale = 0.15 if ball.speed > 100 else 0.1
                arrow_end = (
                    int(ball.last_center[0] + ball.velocity[0] * scale),
                    int(ball.last_center[1] + ball.velocity[1] * scale)
                )
                cv2.arrowedLine(frame, ball.last_center, arrow_end, center_color, 2, tipLength=0.3)
            
            # Draw predicted position (larger circle for fast objects)
            if show_prediction and ball.predicted_position is not None:
                x_pred, y_pred = ball.predicted_position
                if 0 <= x_pred < frame.shape[1] and 0 <= y_pred < frame.shape[0]:
                    pred_radius = max(8, int(ball.speed / 20))
                    cv2.circle(frame, ball.predicted_position, pred_radius, (255, 255, 0), 2)
                    cv2.line(frame, ball.last_center, ball.predicted_position, 
                            (255, 255, 0), 1, cv2.LINE_AA)
            
            # Draw speed indicator
            if show_kalman_state and ball.speed > 10:
                speed_text = f"{ball.speed:.0f}px/s"
                cv2.putText(frame, speed_text, 
                           (ball.last_center[0] + 10, ball.last_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, center_color, 1, cv2.LINE_AA)
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        active_tracks = len(self.tracked_balls)
        predicted_tracks = sum(1 for ball in self.tracked_balls.values() if ball.frames_missing > 0)
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': active_tracks,
            'predicted_tracks': predicted_tracks,
            'detected_tracks': active_tracks - predicted_tracks,
            'total_detections': self.total_detections,
            'current_detections': self.detection_count,
            'next_id': self.next_id
        }


def draw_advanced_hud(frame: np.ndarray, tracker: AdvancedYOLOBallTracker, fps: float):
    """Draw enhanced HUD"""
    stats = tracker.get_statistics()
    
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Active: {stats['active_tracks']} ({stats['detected_tracks']} detected, {stats['predicted_tracks']} predicted)",
        f"Frame: {stats['frame_count']}",
        f"Total Detections: {stats['total_detections']}",
    ]
    
    # Draw HUD background
    hud_height = len(hud_lines) * 25 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, hud_height), (0, 0, 0), -1)
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
            status = "PREDICTING" if ball.frames_missing > 0 else "TRACKING"
            info = f"Ball {track_id}: {status} Speed={ball.speed:.1f}px/s"
            cv2.putText(frame, info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, ball.color, 2, cv2.LINE_AA)
            y_offset += 20


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced YOLO Ball Tracking (Fast Motion)")
    parser.add_argument('-v', '--video', help="Path to video file")
    parser.add_argument('-m', '--model', default='yolov8n.pt', help="YOLO model")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (lower for fast objects)")
    parser.add_argument('--max-distance', type=float, default=150.0, help="Max matching distance")
    parser.add_argument('--no-blur-reduction', action='store_true', help="Disable blur reduction")
    parser.add_argument('--hud', action='store_true', help="Display HUD")
    parser.add_argument('--no-trails', action='store_true', help="Disable trails")
    parser.add_argument('--display-fps', action='store_true', help="Print FPS")
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("[ERROR] Ultralytics YOLO not installed")
        return
    
    # Initialize video
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"[INFO] Using video file: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("[INFO] Using webcam")
    
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return
    
    # Initialize tracker
    tracker = AdvancedYOLOBallTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
        max_distance=args.max_distance,
        reduce_blur=not args.no_blur_reduction
    )
    
    # FPS calculation
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0
    
    print("[INFO] Advanced tracking active... Press 'q' to quit")
    print("[INFO] Features: Kalman filtering, motion prediction, blur reduction")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track balls
        frame_time = time.time()
        tracked_balls = tracker.update(frame, frame_time)
        
        # Draw tracking
        tracker.draw_tracks(frame, show_trails=not args.no_trails)
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            if args.display_fps:
                stats = tracker.get_statistics()
                print(f"[INFO] FPS: {fps:.2f} | Active: {stats['active_tracks']} | Predicted: {stats['predicted_tracks']}")
        
        # Draw HUD
        if args.hud:
            draw_advanced_hud(frame, tracker, fps)
        
        # Display
        cv2.imshow('Advanced YOLO Ball Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    stats = tracker.get_statistics()
    print("\n[INFO] Tracking Complete!")
    print(f"  Total Frames: {stats['frame_count']}")
    print(f"  Total Detections: {stats['total_detections']}")
    print(f"  Unique Balls Tracked: {stats['next_id']}")
    print(f"  Final FPS: {fps:.2f}")


if __name__ == '__main__':
    main()

