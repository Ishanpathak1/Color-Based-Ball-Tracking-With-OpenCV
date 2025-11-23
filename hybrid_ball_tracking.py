#!/usr/bin/env python3
# HYBRID BALL TRACKING - YOLO Detection + Advanced Color Tracking
# Best of both worlds: YOLO's robust detection + Color-based precision tracking

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import tkinter as tk
from tkinter import simpledialog

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


def str_to_hsv(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be provided as H,S,V (e.g. 29,86,6)")
    try:
        values = tuple(int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers") from exc
    for component in values:
        if component < 0 or component > 255:
            raise argparse.ArgumentTypeError("HSV values must be between 0 and 255")
    return values


def str_to_bgr(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Colors must be provided as B,G,R (e.g. 0,0,255)")
    try:
        values = tuple(int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color components must be integers") from exc
    for component in values:
        if component < 0 or component > 255:
            raise argparse.ArgumentTypeError("Color components must be between 0 and 255")
    return values


def sanitize_bounds(lower, upper):
    lower = np.clip(lower, 0, 255)
    upper = np.clip(upper, 0, 255)
    for idx in range(3):
        if lower[idx] >= upper[idx]:
            if lower[idx] >= 255:
                lower[idx] = 254
                upper[idx] = 255
            else:
                upper[idx] = min(255, lower[idx] + 1)
    return lower, upper


def brighten_color(color: np.ndarray, alpha: float = 1.2, beta: float = 25.0) -> np.ndarray:
    color = np.array(color, dtype=np.float32)
    return np.clip(color * alpha + beta, 0, 255)


@dataclass
class HybridTarget:
    """Hybrid target using both YOLO and color tracking"""
    track_id: int
    name: str
    color: Tuple[int, int, int]
    buffer_len: int = 64
    
    # Color-based tracking (optional, for refinement)
    use_color_tracking: bool = False
    lower_hsv: Optional[np.ndarray] = None
    upper_hsv: Optional[np.ndarray] = None
    
    # Detection source tracking
    last_detection_method: str = "none"  # "yolo", "color", or "predicted"
    
    # History
    pts: deque = field(init=False)
    timestamps: deque = field(init=False)
    confidences: deque = field(init=False)
    
    # Current state
    last_center: Optional[Tuple[int, int]] = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_confidence: float = 0.0
    last_radius: float = 0.0
    smoothed_center: Optional[np.ndarray] = None
    
    # Motion tracking
    velocity: Optional[np.ndarray] = None
    speed: float = 0.0
    predicted_point: Optional[Tuple[int, int]] = None
    
    # Track management
    frames_missing: int = 0
    active: bool = True
    
    def __post_init__(self):
        self.pts = deque(maxlen=self.buffer_len)
        self.timestamps = deque(maxlen=self.buffer_len)
        self.confidences = deque(maxlen=self.buffer_len)
        if self.lower_hsv is not None:
            self.lower_hsv = np.array(self.lower_hsv, dtype=np.float32)
        if self.upper_hsv is not None:
            self.upper_hsv = np.array(self.upper_hsv, dtype=np.float32)
            if self.lower_hsv is not None:
                self.lower_hsv, self.upper_hsv = sanitize_bounds(self.lower_hsv, self.upper_hsv)
    
    def set_buffer_len(self, buffer_len: int):
        if buffer_len != self.buffer_len:
            self.buffer_len = buffer_len
            self.pts = deque(list(self.pts), maxlen=buffer_len)
            self.timestamps = deque(list(self.timestamps), maxlen=buffer_len)
            self.confidences = deque(list(self.confidences), maxlen=buffer_len)
    
    def update(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int],
               confidence: float, timestamp: float, method: str, smoothing: float = 0.2):
        """Update target state"""
        self.last_detection_method = method
        
        # Apply smoothing
        if smoothing > 0:
            current_center = np.array(center, dtype=np.float32)
            if self.smoothed_center is None:
                self.smoothed_center = current_center
            else:
                self.smoothed_center = smoothing * current_center + (1.0 - smoothing) * self.smoothed_center
            center = (int(self.smoothed_center[0]), int(self.smoothed_center[1]))
        
        self.pts.appendleft(center)
        self.timestamps.appendleft(timestamp)
        self.confidences.appendleft(confidence)
        
        self.last_center = center
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.frames_missing = 0
        
        # Calculate radius from bbox
        if bbox is not None:
            self.last_radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0
        
        # Calculate velocity
        self._calculate_velocity()
    
    def _calculate_velocity(self):
        """Calculate velocity from position history"""
        self.velocity = None
        self.speed = 0.0
        
        if len(self.pts) < 2:
            return
        
        for i in range(1, min(5, len(self.pts))):
            if self.pts[i] is None or self.timestamps[i] is None:
                continue
            if self.timestamps[0] is None:
                continue
            
            dt = self.timestamps[0] - self.timestamps[i]
            if dt > 1e-6:
                displacement = np.array(self.pts[0], dtype=np.float32) - np.array(self.pts[i], dtype=np.float32)
                self.velocity = displacement / dt
                self.speed = float(np.linalg.norm(self.velocity))
                
                # Predict next position
                if self.last_center is not None:
                    pred = np.array(self.last_center, dtype=np.float32) + self.velocity * 0.1
                    self.predicted_point = (int(pred[0]), int(pred[1]))
                break
    
    def mark_missing(self):
        """Mark that target wasn't detected this frame"""
        self.frames_missing += 1
        self.last_detection_method = "predicted"
    
    def is_active(self, max_missing: int = 30) -> bool:
        """Check if target should still be tracked"""
        return self.frames_missing < max_missing


class HybridTracker:
    """Hybrid tracker combining YOLO and color-based detection"""
    
    BALL_CLASSES = {'sports ball': 32, 'baseball': 36, 'frisbee': 29}
    COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255)]
    
    def __init__(self, use_yolo: bool = True, use_color: bool = True,
                 yolo_model: str = 'yolov8n.pt', yolo_conf: float = 0.25,
                 max_distance: float = 120.0):
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.use_color = use_color
        self.max_distance = max_distance
        
        # YOLO setup
        self.yolo_model = None
        if self.use_yolo:
            print(f"[INFO] Loading YOLO model: {yolo_model}")
            self.yolo_model = YOLO(yolo_model)
            self.yolo_conf = yolo_conf
        
        # Tracking state
        self.targets: Dict[int, HybridTarget] = {}
        self.next_id = 0
        self.frame_count = 0
        
        # Statistics
        self.yolo_detections = 0
        self.color_detections = 0
        self.hybrid_detections = 0
    
    def detect_yolo(self, frame: np.ndarray) -> List[Tuple]:
        """Detect balls using YOLO"""
        if not self.use_yolo or self.yolo_model is None:
            return []
        
        results = self.yolo_model(frame, conf=self.yolo_conf, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                
                if class_name in self.BALL_CLASSES or 'ball' in class_name.lower():
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    detections.append((center, bbox, conf, "yolo"))
                    self.yolo_detections += 1
        
        return detections
    
    def detect_color(self, frame: np.ndarray, hsv: np.ndarray, 
                    target: HybridTarget, kernel: np.ndarray) -> Optional[Tuple]:
        """Detect ball using color-based method for specific target"""
        if not self.use_color or not target.use_color_tracking:
            return None
        if target.lower_hsv is None or target.upper_hsv is None:
            return None
        
        # Create mask
        mask = cv2.inRange(hsv, target.lower_hsv.astype(np.uint8), target.upper_hsv.astype(np.uint8))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_center = None
        best_bbox = None
        best_score = -1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 3:
                continue
            
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            
            # Prefer detections near last known position or predicted position
            score = area
            if target.predicted_point is not None:
                dist = np.linalg.norm(np.array(center) - np.array(target.predicted_point))
                score = area / (1 + dist / 50.0)
            elif target.last_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(target.last_center))
                score = area / (1 + dist / 50.0)
            
            if score > best_score:
                best_score = score
                best_center = center
                x1, y1 = int(x - radius), int(y - radius)
                x2, y2 = int(x + radius), int(y + radius)
                best_bbox = (x1, y1, x2, y2)
        
        if best_center is not None:
            confidence = min(best_score / 500.0, 1.0)
            self.color_detections += 1
            return (best_center, best_bbox, confidence, "color")
        
        return None
    
    def match_detection_to_track(self, center: Tuple[int, int]) -> Optional[int]:
        """Match detection to existing track"""
        best_match = None
        min_distance = self.max_distance
        
        for track_id, target in self.targets.items():
            # Try predicted position first
            if target.predicted_point is not None:
                dist = np.linalg.norm(np.array(center) - np.array(target.predicted_point))
                if dist < min_distance:
                    min_distance = dist
                    best_match = track_id
                    continue
            
            # Fall back to last position
            if target.last_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(target.last_center))
                if dist < min_distance:
                    min_distance = dist
                    best_match = track_id
        
        return best_match
    
    def update(self, frame: np.ndarray, hsv: np.ndarray, timestamp: float,
               kernel: np.ndarray, smoothing: float = 0.2) -> Dict[int, HybridTarget]:
        """Update tracker with new frame - HYBRID APPROACH"""
        self.frame_count += 1
        
        # Step 1: Get YOLO detections
        yolo_detections = self.detect_yolo(frame) if self.use_yolo else []
        
        # Step 2: Get color-based detections for existing targets
        color_detections = []
        for target_id, target in self.targets.items():
            color_det = self.detect_color(frame, hsv, target, kernel)
            if color_det is not None:
                color_detections.append((target_id, color_det))
        
        # Step 3: Match YOLO detections to existing tracks
        matched_tracks = set()
        
        for center, bbox, conf, method in yolo_detections:
            track_id = self.match_detection_to_track(center)
            
            if track_id is not None and track_id not in matched_tracks:
                # Update existing track with YOLO detection
                self.targets[track_id].update(center, bbox, conf, timestamp, method, smoothing)
                matched_tracks.add(track_id)
            elif track_id is None:
                # Create new track
                color = self.COLORS[self.next_id % len(self.COLORS)]
                new_target = HybridTarget(
                    track_id=self.next_id,
                    name=f"Ball {self.next_id}",
                    color=color,
                    buffer_len=64
                )
                new_target.update(center, bbox, conf, timestamp, method, smoothing)
                self.targets[self.next_id] = new_target
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # Step 4: Update unmatched tracks with color detection if available
        for target_id, color_det in color_detections:
            if target_id not in matched_tracks:
                center, bbox, conf, method = color_det
                self.targets[target_id].update(center, bbox, conf, timestamp, method, smoothing)
                matched_tracks.add(target_id)
                self.hybrid_detections += 1
        
        # Step 5: Mark remaining tracks as missing (use prediction)
        for track_id, target in list(self.targets.items()):
            if track_id not in matched_tracks:
                target.mark_missing()
        
        # Step 6: Remove inactive tracks
        self.targets = {
            tid: target for tid, target in self.targets.items()
            if target.is_active(max_missing=45)
        }
        
        return self.targets
    
    def draw_tracks(self, frame: np.ndarray, show_trails: bool = True,
                   show_velocity: bool = True, show_method: bool = True):
        """Draw all tracks"""
        for track_id, target in self.targets.items():
            if target.last_center is None:
                continue
            
            # Color based on detection method
            if show_method:
                if target.last_detection_method == "yolo":
                    draw_color = target.color
                elif target.last_detection_method == "color":
                    draw_color = tuple(int(c * 0.7) for c in target.color)
                else:  # predicted
                    draw_color = tuple(int(c * 0.4) for c in target.color)
            else:
                draw_color = target.color
            
            # Draw bounding box
            if target.last_bbox is not None and target.frames_missing < 3:
                x1, y1, x2, y2 = target.last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                
                # Label with method
                method_label = target.last_detection_method.upper()
                label = f"ID:{track_id} {method_label} {target.last_confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, draw_color, 2, cv2.LINE_AA)
            
            # Draw center
            cv2.circle(frame, target.last_center, 5, draw_color, -1)
            
            # Draw trail
            if show_trails and len(target.pts) > 1:
                for i in range(1, len(target.pts)):
                    if target.pts[i] is None:
                        continue
                    thickness = max(1, int(np.sqrt(target.buffer_len / float(i + 1)) * 2.5))
                    alpha = max(0.2, 1.0 - (i / len(target.pts)))
                    color = tuple(int(c * alpha) for c in target.color)
                    cv2.line(frame, target.pts[i-1], target.pts[i], color, thickness)
            
            # Draw velocity arrow
            if show_velocity and target.velocity is not None and target.speed > 5:
                arrow_end = (
                    int(target.last_center[0] + target.velocity[0] * 0.1),
                    int(target.last_center[1] + target.velocity[1] * 0.1)
                )
                cv2.arrowedLine(frame, target.last_center, arrow_end, draw_color, 2, tipLength=0.3)
            
            # Draw prediction
            if target.predicted_point is not None:
                x_pred, y_pred = target.predicted_point
                if 0 <= x_pred < frame.shape[1] and 0 <= y_pred < frame.shape[0]:
                    cv2.circle(frame, target.predicted_point, 8, (255, 255, 0), 2)
                    cv2.line(frame, target.last_center, target.predicted_point,
                            (255, 255, 0), 1, cv2.LINE_AA)
    
    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        yolo_count = sum(1 for t in self.targets.values() if t.last_detection_method == "yolo")
        color_count = sum(1 for t in self.targets.values() if t.last_detection_method == "color")
        pred_count = sum(1 for t in self.targets.values() if t.last_detection_method == "predicted")
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.targets),
            'yolo_detections': yolo_count,
            'color_detections': color_count,
            'predicted_tracks': pred_count,
            'total_yolo': self.yolo_detections,
            'total_color': self.color_detections,
            'total_hybrid': self.hybrid_detections
        }


def draw_hud(frame: np.ndarray, tracker: HybridTracker, fps: float):
    """Draw HUD overlay"""
    stats = tracker.get_stats()
    
    hud_lines = [
        f"FPS: {fps:.1f} | Mode: HYBRID (YOLO + Color)",
        f"Active: {stats['active_tracks']} (YOLO:{stats['yolo_detections']} COLOR:{stats['color_detections']} PRED:{stats['predicted_tracks']})",
        f"Frame: {stats['frame_count']} | Total: YOLO={stats['total_yolo']} Color={stats['total_color']} Hybrid={stats['total_hybrid']}",
    ]
    
    overlay = frame.copy()
    hud_height = len(hud_lines) * 25 + 20
    cv2.rectangle(overlay, (10, 10), (650, hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    y_offset = 30
    for line in hud_lines:
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 25
    
    # Ball details
    y_offset = hud_height + 25
    for track_id, target in tracker.targets.items():
        method_color = {
            'yolo': (0, 255, 255),
            'color': (255, 128, 0),
            'predicted': (128, 128, 128)
        }.get(target.last_detection_method, (255, 255, 255))
        
        info = f"Ball {track_id}: {target.last_detection_method.upper()} Speed={target.speed:.1f}px/s"
        cv2.putText(frame, info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.45, method_color, 1, cv2.LINE_AA)
        y_offset += 20


def main():
    parser = argparse.ArgumentParser(description="Hybrid Ball Tracking (YOLO + Color)")
    parser.add_argument('-v', '--video', help="Path to video file")
    parser.add_argument('--yolo-only', action='store_true', help="Use YOLO only (no color refinement)")
    parser.add_argument('--color-only', action='store_true', help="Use color only (no YOLO)")
    parser.add_argument('--model', default='yolov8n.pt', help="YOLO model path")
    parser.add_argument('--conf', type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument('--max-distance', type=float, default=120.0, help="Max matching distance")
    parser.add_argument('--smoothing', type=float, default=0.2, help="Position smoothing factor")
    parser.add_argument('--hud', action='store_true', help="Display HUD")
    parser.add_argument('--display-fps', action='store_true', help="Print FPS")
    parser.add_argument('--no-trails', action='store_true', help="Disable trails")
    parser.add_argument('--buffer', type=int, default=64, help="Trail buffer size")
    
    args = parser.parse_args()
    
    # Determine mode
    use_yolo = not args.color_only
    use_color = not args.yolo_only
    
    if args.yolo_only and args.color_only:
        print("[ERROR] Cannot use both --yolo-only and --color-only")
        return
    
    if not YOLO_AVAILABLE and use_yolo:
        print("[WARNING] YOLO not available, falling back to color-only mode")
        use_yolo = False
        use_color = True
    
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
    tracker = HybridTracker(
        use_yolo=use_yolo,
        use_color=use_color,
        yolo_model=args.model,
        yolo_conf=args.conf,
        max_distance=args.max_distance
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # FPS tracking
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0
    
    mode_name = "HYBRID (YOLO + Color)" if (use_yolo and use_color) else ("YOLO ONLY" if use_yolo else "COLOR ONLY")
    print(f"[INFO] Starting {mode_name} tracking... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize
        frame = imutils.resize(frame, width=800)
        
        # Prepare HSV
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Track
        frame_time = time.time()
        tracked = tracker.update(frame, hsv, frame_time, kernel, args.smoothing)
        
        # Draw
        tracker.draw_tracks(frame, show_trails=not args.no_trails)
        
        # FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            if args.display_fps:
                stats = tracker.get_stats()
                print(f"[INFO] FPS: {fps:.2f} | Active: {stats['active_tracks']} | YOLO:{stats['yolo_detections']} COLOR:{stats['color_detections']}")
        
        # HUD
        if args.hud:
            draw_hud(frame, tracker, fps)
        
        # Display
        cv2.imshow('Hybrid Ball Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    stats = tracker.get_stats()
    print("\n[INFO] Tracking Complete!")
    print(f"  Total Frames: {stats['frame_count']}")
    print(f"  YOLO Detections: {stats['total_yolo']}")
    print(f"  Color Detections: {stats['total_color']}")
    print(f"  Hybrid Assists: {stats['total_hybrid']}")
    print(f"  Final FPS: {fps:.2f}")


if __name__ == '__main__':
    main()

