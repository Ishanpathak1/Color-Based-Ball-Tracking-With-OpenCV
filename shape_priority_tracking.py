# USAGE
# python shape_priority_tracking.py
# python shape_priority_tracking.py --video ball_tracking_example.mp4
# python shape_priority_tracking.py --display-mask --fps

# Shape-Priority Ball Tracking
# Primary filter: Shape (circularity, roundness)
# Secondary filter: Color (HSV range as hint)
# Perfect for soft balls that can be pressed/deformed to stop drawing

import numpy as np
import argparse
import cv2
import imutils
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from imutils.video import VideoStream

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

@dataclass
class ShapeTracker:
	name: str
	min_circularity: float  # 0.7 = pretty round, 0.85 = very round
	min_radius: float
	max_radius: float
	color_lower: Optional[np.ndarray] = None  # Optional color hint
	color_upper: Optional[np.ndarray] = None
	buffer_len: int = 64
	trail_color: Tuple[int, int, int] = (0, 255, 0)
	
	# Drawing state
	is_drawing: bool = field(default=False, init=False)
	pts: deque = field(init=False)
	timestamps: deque = field(init=False)
	last_center: Optional[Tuple[int, int]] = None
	last_radius: float = 0.0
	last_circularity: float = 0.0
	smoothed_center: Optional[np.ndarray] = None
	
	def __post_init__(self):
		self.pts = deque(maxlen=self.buffer_len)
		self.timestamps = deque(maxlen=self.buffer_len)
		if self.color_lower is not None:
			self.color_lower = np.array(self.color_lower, dtype=np.uint8)
		if self.color_upper is not None:
			self.color_upper = np.array(self.color_upper, dtype=np.uint8)


def compute_circularity(contour) -> float:
	"""
	Compute circularity using 4π·area/perimeter²
	Returns 1.0 for perfect circle, lower for non-circular shapes
	"""
	area = cv2.contourArea(contour)
	perimeter = cv2.arcLength(contour, True)
	if perimeter == 0:
		return 0.0
	circularity = (4 * np.pi * area) / (perimeter * perimeter)
	return min(circularity, 1.0)  # Cap at 1.0


def compute_extent(contour) -> float:
	"""
	Compute extent = contour_area / bounding_box_area
	Returns how much of the bounding box is filled (1.0 for circle/square)
	"""
	area = cv2.contourArea(contour)
	_, _, w, h = cv2.boundingRect(contour)
	if w == 0 or h == 0:
		return 0.0
	bbox_area = w * h
	return area / bbox_area


def compute_aspect_ratio(contour) -> float:
	"""
	Compute aspect ratio of bounding rectangle
	Returns value close to 1.0 for circles
	"""
	_, _, w, h = cv2.boundingRect(contour)
	if h == 0:
		return 0.0
	return float(w) / float(h)


def detect_circular_objects(
	frame: np.ndarray,
	tracker: ShapeTracker,
	frame_time: float,
	smoothing_factor: float = 0.2,
	color_weight: float = 0.3,
	min_area: float = 100.0,
) -> Optional[np.ndarray]:
	"""
	Detect circular objects using shape as primary filter
	
	Parameters:
	- color_weight: How much to weigh color matching (0.0 = ignore color, 1.0 = heavily favor)
	"""
	# Convert to grayscale and HSV
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	blurred_gray = cv2.GaussianBlur(gray, (11, 11), 0)
	
	# Start with a clean slate - only detect isolated bright/dark spots
	combined = np.zeros(gray.shape, dtype=np.uint8)
	
	# Method 1: Detect dark objects on light background
	_, thresh_dark = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	# Method 2: Detect bright objects on dark background  
	_, thresh_bright = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# Method 3: Edge detection (more conservative)
	edges = cv2.Canny(blurred_gray, 50, 150)
	
	# Only keep small to medium isolated regions (not huge blobs like hands/laptops)
	kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	
	# Process dark objects
	thresh_dark = cv2.morphologyEx(thresh_dark, cv2.MORPH_OPEN, kernel_small, iterations=2)
	thresh_dark = cv2.morphologyEx(thresh_dark, cv2.MORPH_CLOSE, kernel_small, iterations=1)
	
	# Process bright objects
	thresh_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_OPEN, kernel_small, iterations=2)
	thresh_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel_small, iterations=1)
	
	# Combine methods
	combined = cv2.bitwise_or(thresh_dark, thresh_bright)
	combined = cv2.bitwise_or(combined, edges)
	
	# Find all contours
	cnts = cv2.findContours(combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	# Color mask as PRIMARY filter when provided
	color_mask = None
	if tracker.color_lower is not None and tracker.color_upper is not None:
		# Create color mask
		color_mask = cv2.inRange(hsv, tracker.color_lower, tracker.color_upper)
		
		# Clean up the color mask
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
		color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
		color_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)
		
		# If color is provided with high weight, use it as the PRIMARY mask
		if color_weight > 0.5:
			# Color is primary - use it directly
			combined = color_mask
		else:
			# Color is hint - boost the combined mask where color matches
			color_boosted = cv2.bitwise_and(combined, color_mask)
			combined = cv2.bitwise_or(combined, color_boosted)
	
	best_candidate = None
	best_score = -1.0
	best_center = None
	best_radius = 0.0
	best_circularity = 0.0
	
	previous_point = tracker.pts[0] if len(tracker.pts) > 0 and tracker.pts[0] is not None else None
	
	# Also try HoughCircles as a fallback detector for perfect circles
	hough_circles = cv2.HoughCircles(
		blurred_gray,
		cv2.HOUGH_GRADIENT,
		dp=1.2,
		minDist=50,
		param1=50,
		param2=30,
		minRadius=int(tracker.min_radius),
		maxRadius=int(tracker.max_radius)
	)
	
	hough_candidates = []
	if hough_circles is not None:
		hough_circles = np.round(hough_circles[0, :]).astype("int")
		for (x, y, r) in hough_circles:
			hough_candidates.append(((x, y), r, 0.95))  # High circularity for Hough circles
	
	for contour in cnts:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue
		
		# CRITICAL: Reject huge areas (hands, laptops, etc.)
		# A ball should have reasonable size, not fill entire frame
		max_reasonable_area = (tracker.max_radius * 2) ** 2 * 2  # Roughly 2x the max circle area
		if area > max_reasonable_area:
			continue
		
		# Shape metrics
		circularity = compute_circularity(contour)
		# Don't filter too early - let scoring decide
		if circularity < max(0.4, tracker.min_circularity - 0.25):  # More lenient early filter
			continue
		
		extent = compute_extent(contour)
		aspect_ratio = compute_aspect_ratio(contour)
		
		# Check if aspect ratio is close to 1.0 (circle-like)
		aspect_score = 1.0 - abs(1.0 - min(aspect_ratio, 1.0/aspect_ratio) - 1.0)
		if aspect_score < 0.3:  # More lenient - allow slightly elongated shapes
			continue
		
		# Get enclosing circle
		(x, y), radius = cv2.minEnclosingCircle(contour)
		if radius < tracker.min_radius or radius > tracker.max_radius:
			continue
		
		# Compute moments for centroid
		M = cv2.moments(contour)
		if M["m00"] == 0:
			continue
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
		# Shape score (primary)
		shape_score = (circularity * 0.6 + extent * 0.2 + aspect_score * 0.2)
		
		# Color score (secondary)
		color_score = 0.5  # Neutral if no color filter
		if color_mask is not None:
			# Check how much of the contour overlaps with color mask
			mask_roi = np.zeros(color_mask.shape, dtype=np.uint8)
			cv2.drawContours(mask_roi, [contour], -1, 255, -1)
			overlap = cv2.bitwise_and(color_mask, mask_roi)
			overlap_ratio = np.count_nonzero(overlap) / max(np.count_nonzero(mask_roi), 1)
			color_score = overlap_ratio
		
		# Temporal consistency (prefer objects close to previous position)
		temporal_score = 1.0
		if previous_point is not None:
			distance = np.linalg.norm(np.array(center) - np.array(previous_point))
			max_expected_distance = 100.0  # pixels
			temporal_score = max(0.0, 1.0 - (distance / max_expected_distance))
		
		# Bonus for Hough circle overlap
		hough_bonus = 0.0
		for (hx, hy), hr, _ in hough_candidates:
			dist_to_hough = np.sqrt((center[0] - hx)**2 + (center[1] - hy)**2)
			if dist_to_hough < 20 and abs(radius - hr) < 15:  # Close to a Hough circle
				hough_bonus = 0.2
				break
		
		# Combined score: shape is most important, color is hint, temporal helps stability
		total_score = (
			shape_score * 0.45 +
			color_score * color_weight +
			temporal_score * (0.55 - color_weight) +
			hough_bonus
		)
		
		if total_score > best_score:
			best_score = total_score
			best_candidate = contour
			best_center = center
			best_radius = radius
			best_circularity = circularity
	
	# If no contours found but Hough found circles, use the best Hough circle
	if best_candidate is None and len(hough_candidates) > 0:
		for (hx, hy), hr, hcirc in hough_candidates:
			center_candidate = (hx, hy)
			
			# Temporal check
			temporal_score = 1.0
			if previous_point is not None:
				distance = np.linalg.norm(np.array(center_candidate) - np.array(previous_point))
				max_expected_distance = 100.0
				temporal_score = max(0.0, 1.0 - (distance / max_expected_distance))
			
			# Color check
			color_score = 0.5
			if color_mask is not None:
				# Sample color at circle location
				mask_roi = np.zeros(color_mask.shape, dtype=np.uint8)
				cv2.circle(mask_roi, center_candidate, int(hr), 255, -1)
				overlap = cv2.bitwise_and(color_mask, mask_roi)
				overlap_ratio = np.count_nonzero(overlap) / max(np.count_nonzero(mask_roi), 1)
				color_score = overlap_ratio
			
			score = hcirc * 0.5 + temporal_score * (0.5 - color_weight) + color_score * color_weight
			
			if score > best_score:
				best_score = score
				best_center = center_candidate
				best_radius = float(hr)
				best_circularity = hcirc
	
	# Update tracker state
	tracked_center = None
	tracker.last_circularity = 0.0
	
	if best_candidate is not None and best_center is not None:
		# Smooth the center position
		if smoothing_factor > 0:
			current_center = np.array(best_center, dtype=np.float32)
			if tracker.smoothed_center is None:
				tracker.smoothed_center = current_center
			else:
				tracker.smoothed_center = (
					smoothing_factor * current_center + 
					(1.0 - smoothing_factor) * tracker.smoothed_center
				)
			tracked_center = (int(tracker.smoothed_center[0]), int(tracker.smoothed_center[1]))
		else:
			tracked_center = best_center
		
		tracker.last_radius = float(best_radius)
		tracker.last_circularity = best_circularity
		
		# Determine if we should draw (ball is round enough)
		# When you press the ball, circularity drops, so it stops drawing
		tracker.is_drawing = best_circularity >= tracker.min_circularity
		
		# Draw detection visualization
		cv2.circle(frame, (int(best_center[0]), int(best_center[1])), int(best_radius), (0, 255, 255), 2)
		cv2.circle(frame, tracked_center, 5, tracker.trail_color, -1)
		
		# Show circularity value
		status = "DRAWING" if tracker.is_drawing else "PAUSED"
		status_color = (0, 255, 0) if tracker.is_drawing else (0, 0, 255)
		cv2.putText(
			frame,
			f"Circ:{best_circularity:.2f} {status}",
			(tracked_center[0] - 50, tracked_center[1] - int(best_radius) - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			status_color,
			2,
		)
		
		# Draw a filled circle indicator when drawing
		if tracker.is_drawing:
			cv2.circle(frame, (20, frame.shape[0] - 20), 10, (0, 255, 0), -1)
		else:
			cv2.circle(frame, (20, frame.shape[0] - 20), 10, (0, 0, 255), -1)
	else:
		tracker.smoothed_center = None
		tracker.is_drawing = False
	
	# Update tracking history
	tracker.last_center = tracked_center
	tracker.pts.appendleft(tracked_center if tracker.is_drawing else None)
	tracker.timestamps.appendleft(frame_time if tracked_center is not None else None)
	
	# Draw trail (only for points where we were drawing)
	for i in range(1, len(tracker.pts)):
		if tracker.pts[i - 1] is None or tracker.pts[i] is None:
			continue
		
		thickness = int(np.sqrt(tracker.buffer_len / float(i + 1)) * 2.5)
		cv2.line(frame, tracker.pts[i - 1], tracker.pts[i], tracker.trail_color, thickness)
	
	# Return the combined detection mask for visualization
	return combined


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
ap.add_argument("--min-circularity", type=float, default=0.65,
	help="minimum circularity (0-1) to draw (0.65=lenient, 0.75=fairly round, 0.85=very round)")
ap.add_argument("--min-radius", type=float, default=5.0,
	help="minimum detected radius (in pixels)")
ap.add_argument("--max-radius", type=float, default=200.0,
	help="maximum detected radius (in pixels)")
ap.add_argument("--min-area", type=float, default=50.0,
	help="minimum contour area to consider")
ap.add_argument("--hsv-lower", type=str_to_hsv,
	help="optional lower HSV bounds as H,S,V for color hint (e.g. 29,86,6)")
ap.add_argument("--hsv-upper", type=str_to_hsv,
	help="optional upper HSV bounds as H,S,V for color hint (e.g. 64,255,255)")
ap.add_argument("--color-weight", type=float, default=0.3,
	help="weight for color matching (0.0=ignore, 1.0=heavily favor) default 0.3")
ap.add_argument("--smoothing", type=float, default=0.25,
	help="exponential smoothing factor for the tracked center (0 disables)")
ap.add_argument("--resize-width", type=int, default=600,
	help="resize width for processing (<=0 keeps native resolution)")
ap.add_argument("--display-mask", action="store_true",
	help="display the edge detection / mask window")
ap.add_argument("--fps", action="store_true",
	help="log frames-per-second information to stdout")
args = vars(ap.parse_args())

# Initialize tracker
tracker = ShapeTracker(
	name="Ball",
	min_circularity=args["min_circularity"],
	min_radius=args["min_radius"],
	max_radius=args["max_radius"],
	color_lower=args["hsv_lower"],
	color_upper=args["hsv_upper"],
	buffer_len=args["buffer"],
	trail_color=(0, 255, 0),  # Green trail
)

# Initialize video stream
if args.get("video"):
	vs = cv2.VideoCapture(args["video"])
else:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# Setup windows
cv2.namedWindow("Shape-Priority Tracking", cv2.WINDOW_NORMAL)
if args["display_mask"]:
	cv2.namedWindow("Detection Mask", cv2.WINDOW_NORMAL)

# FPS tracking
frame_counter = 0
fps_checkpoint = time.time()

print("[INFO] Shape-Priority Ball Tracking Started")
print("[INFO] Primary filter: Shape (circularity)")
print("[INFO] Press the ball to deform it -> stops drawing")
print("[INFO] Release the ball to restore shape -> resumes drawing")
print("[INFO] Press 'q' to quit, 'c' to clear trail")

while True:
	# Grab frame
	if args.get("video"):
		ret, frame = vs.read()
		if not ret:
			break
	else:
		frame = vs.read()
	
	if frame is None:
		break
	
	# Resize if needed
	if args["resize_width"] > 0:
		frame = imutils.resize(frame, width=args["resize_width"])
	
	frame_time = time.time()
	
	# Detect and track
	mask = detect_circular_objects(
		frame,
		tracker,
		frame_time,
		smoothing_factor=args["smoothing"],
		color_weight=args["color_weight"],
		min_area=args["min_area"],
	)
	
	# Add HUD
	status_color = (0, 255, 0) if tracker.is_drawing else (0, 0, 255)
	status_text = "DRAWING (Ball is Round)" if tracker.is_drawing else "PAUSED (Ball is Pressed/Deformed)"
	cv2.putText(
		frame,
		status_text,
		(10, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		status_color,
		2,
	)
	
	if tracker.last_center is not None:
		info_text = f"Center: {tracker.last_center} | Radius: {tracker.last_radius:.1f}px | Circularity: {tracker.last_circularity:.2f}"
		cv2.putText(
			frame,
			info_text,
			(10, 60),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255),
			1,
		)
	
	# FPS calculation
	frame_counter += 1
	if args["fps"] and frame_counter % 30 == 0:
		now = time.time()
		elapsed = max(now - fps_checkpoint, 1e-6)
		fps = 30.0 / elapsed
		print(f"[INFO] approx FPS: {fps:.2f}")
		fps_checkpoint = now
	
	# Display
	cv2.imshow("Shape-Priority Tracking", frame)
	if args["display_mask"] and mask is not None:
		cv2.imshow("Detection Mask", mask)
	
	# Keyboard controls
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key == ord("c"):
		tracker.pts.clear()
		tracker.timestamps.clear()
		print("[INFO] Trail cleared")

# Cleanup
if args.get("video"):
	vs.release()
else:
	vs.stop()

cv2.destroyAllWindows()
print("[INFO] Tracking stopped")

