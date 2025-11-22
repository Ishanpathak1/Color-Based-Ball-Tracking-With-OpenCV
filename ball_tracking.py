# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py
# python ball_tracking.py --picamera 1
# python ball_tracking.py --picamera2 1 --display-mask --fps
# python ball_tracking.py --display-mask --fps  # auto-detects PiCamera2 on Raspberry Pi

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import simpledialog
try:
	from picamera2 import Picamera2
except ImportError:
	Picamera2 = None


def str_to_hsv(value):
	parts = value.split(",")
	if len(parts) != 3:
		raise argparse.ArgumentTypeError(
			"HSV values must be provided as H,S,V (e.g. 29,86,6)"
		)
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
		raise argparse.ArgumentTypeError(
			"Colors must be provided as B,G,R (e.g. 0,0,255)"
		)
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

@dataclass
class ColorTarget:
	name: str
	lower: np.ndarray
	upper: np.ndarray
	trail_color: np.ndarray
	buffer_len: int
	active: bool = True
	pts: deque = field(init=False)
	timestamps: deque = field(init=False)
	smoothed_center: Optional[np.ndarray] = None
	last_speed: float = 0.0
	last_velocity: Optional[np.ndarray] = None
	last_radius: float = 0.0
	predicted_point: Optional[Tuple[int, int]] = None
	last_center: Optional[Tuple[int, int]] = None

	def __post_init__(self):
		self.lower = np.array(self.lower, dtype=np.float32)
		self.upper = np.array(self.upper, dtype=np.float32)
		self.lower, self.upper = sanitize_bounds(self.lower, self.upper)
		self.trail_color = np.array(self.trail_color, dtype=np.float32)
		self.pts = deque(maxlen=self.buffer_len)
		self.timestamps = deque(maxlen=self.buffer_len)

	def set_buffer_len(self, buffer_len: int) -> None:
		if buffer_len != self.buffer_len:
			self.buffer_len = buffer_len
			self.pts = deque(list(self.pts), maxlen=buffer_len)
			self.timestamps = deque(list(self.timestamps), maxlen=buffer_len)

	def info_label(self) -> str:
		state = "ON" if self.active else "OFF"
		return f"{self.name} [{state}]"

def brighten_color(color: np.ndarray, alpha: float = 1.2, beta: float = 25.0) -> np.ndarray:
	color = np.array(color, dtype=np.float32)
	return np.clip(color * alpha + beta, 0, 255)


def bgr_to_hex(color: np.ndarray) -> str:
	b, g, r = [int(np.clip(c, 0, 255)) for c in color]
	return f"#{r:02x}{g:02x}{b:02x}"


def next_target_name(existing_names: List[str], base: str = "Ball") -> str:
	idx = 1
	candidate = base
	while candidate in existing_names:
		idx += 1
		candidate = f"{base} {idx}"
	return candidate


def sample_color_from_point(
	hsv_frame: np.ndarray,
	bgr_frame: np.ndarray,
	point: Tuple[int, int],
	sample_radius: int = 12,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
	h, w = hsv_frame.shape[:2]
	x, y = point
	x0 = max(x - sample_radius, 0)
	y0 = max(y - sample_radius, 0)
	x1 = min(x + sample_radius, w - 1)
	y1 = min(y + sample_radius, h - 1)
	roi_hsv = hsv_frame[y0:y1 + 1, x0:x1 + 1]
	roi_bgr = bgr_frame[y0:y1 + 1, x0:x1 + 1]
	if roi_hsv.size == 0 or roi_bgr.size == 0:
		return None
	roi_flat = roi_hsv.reshape(-1, 3)
	lower = np.percentile(roi_flat, 10, axis=0)
	upper = np.percentile(roi_flat, 90, axis=0)
	lower = np.clip(lower - 10, 0, 255)
	upper = np.clip(upper + 10, 0, 255)
	mean_bgr = roi_bgr.reshape(-1, 3).mean(axis=0)
	return lower.astype(np.float32), upper.astype(np.float32), mean_bgr.astype(np.float32)


def reset_target_history(target: ColorTarget) -> None:
	target.pts.clear()
	target.timestamps.clear()
	target.smoothed_center = None
	target.last_center = None
	target.last_velocity = None
	target.last_speed = 0.0
	target.predicted_point = None
	target.last_radius = 0.0


class ControlPanel:
	THEME_LOOKUP = {
		"DarkBlue3": ("#111b2f", "#f0f0f0"),
		"Dark": ("#1f232a", "#f0f0f0"),
		"Light": ("#f4f5f9", "#202020"),
	}

	def __init__(self, theme: str = "DarkBlue3"):
		bg_color, fg_color = self.THEME_LOOKUP.get(theme, ("#111b2f", "#f0f0f0"))
		self.root = tk.Tk()
		self.root.title("Ball Tracker Control")
		self.root.configure(bg=bg_color)
		self.root.resizable(False, False)
		self.root.protocol("WM_DELETE_WINDOW", self._on_close)

		self._bg_color = bg_color
		self._fg_color = fg_color
		self.events: List[Tuple[str, Optional[int]]] = []
		self.target_buttons: List[tk.Button] = []
		self.selected_index: Optional[int] = None
		self._signature: Optional[Tuple] = None
		self._closed = False

		self.header = tk.Label(
			self.root,
			text="Ball Tracker Control",
			font=("Helvetica", 14, "bold"),
			bg=bg_color,
			fg=fg_color,
			anchor="w",
			padx=8,
			pady=6,
		)
		self.header.pack(fill="x")

		self.targets_container = tk.Frame(self.root, bg=bg_color, padx=8, pady=4)
		self.targets_container.pack(fill="both", expand=True)

		self.button_frame = tk.Frame(self.root, bg=bg_color, padx=8, pady=4)
		self.button_frame.pack(fill="x")

		self.pick_button = tk.Button(self.button_frame, text="Pick From Frame", width=16, command=self._on_pick)
		self.pick_button.pack(side="left", padx=2)

		self.clear_button = tk.Button(self.button_frame, text="Clear Trails", width=12, command=self._on_clear)
		self.clear_button.pack(side="left", padx=2)

		self.remove_button = tk.Button(self.button_frame, text="Remove Selected", width=16, command=self._on_remove)
		self.remove_button.pack(side="left", padx=2)

		self.close_button = tk.Button(self.button_frame, text="Close Panel", width=12, command=self._on_close)
		self.close_button.pack(side="left", padx=2)

		self.status_label = tk.Label(self.root, text="", bg=bg_color, fg=fg_color, anchor="w", padx=8, pady=6)
		self.status_label.pack(fill="x")

	def refresh_targets(self, targets: List[ColorTarget], selected_index: Optional[int]) -> None:
		signature = tuple((t.name, t.active, tuple(np.round(t.trail_color, 2))) for t in targets)
		if signature != self._signature:
			self._rebuild_targets(targets)
			self._signature = signature
		self.selected_index = selected_index
		for idx, button in enumerate(self.target_buttons):
			target = targets[idx]
			bg_hex, fg_hex = self._colors_for_target(target)
			button.configure(text=target.info_label(), bg=bg_hex, fg=fg_hex)
			relief = tk.SUNKEN if selected_index == idx else tk.RAISED
			button.configure(relief=relief)

	def set_status(self, text: str) -> None:
		self.status_label.configure(text=text)

	def set_pick_mode(self, awaiting: bool) -> None:
		self.pick_button.configure(state=tk.DISABLED if awaiting else tk.NORMAL)

	def poll(self) -> bool:
		if self._closed:
			self.destroy()
			return False
		try:
			self.root.update_idletasks()
			self.root.update()
		except tk.TclError:
			self._closed = True
			return False
		return True

	def pop_events(self) -> List[Tuple[str, Optional[int]]]:
		events = self.events[:]
		self.events.clear()
		return events

	def destroy(self) -> None:
		if not self._closed:
			self._closed = True
		try:
			self.root.destroy()
		except tk.TclError:
			pass

	def is_open(self) -> bool:
		return not self._closed

	def _colors_for_target(self, target: ColorTarget) -> Tuple[str, str]:
		if target.active:
			color = brighten_color(target.trail_color, alpha=1.12, beta=12.0)
			fg = "#ffffff"
		else:
			color = np.array((80.0, 80.0, 80.0), dtype=np.float32)
			fg = "#bbbbbb"
		return bgr_to_hex(color), fg

	def _rebuild_targets(self, targets: List[ColorTarget]) -> None:
		for widget in self.targets_container.winfo_children():
			widget.destroy()
		self.target_buttons = []
		if not targets:
			tk.Label(self.targets_container, text="No colors added yet", bg=self._bg_color, fg="#cccccc").pack(fill="x", pady=4)
			return
		for idx, target in enumerate(targets):
			bg_hex, fg_hex = self._colors_for_target(target)
			button = tk.Button(
				self.targets_container,
				text=target.info_label(),
				width=28,
				anchor="w",
				bg=bg_hex,
				fg=fg_hex,
				relief=tk.RAISED,
				command=lambda i=idx: self._on_target_click(i),
				padx=10,
				pady=4,
			)
			button.pack(fill="x", pady=2)
			self.target_buttons.append(button)

	def _on_target_click(self, index: int) -> None:
		self.selected_index = index
		self.events.append(("toggle_target", index))

	def _on_pick(self) -> None:
		self.events.append(("pick", None))

	def _on_clear(self) -> None:
		self.events.append(("clear", None))

	def _on_remove(self) -> None:
		self.events.append(("remove", None))

	def _on_close(self) -> None:
		if not self._closed:
			self._closed = True
			self.events.append(("close", None))


def process_target(
	target: ColorTarget,
	hsv_frame: np.ndarray,
	bgr_frame: np.ndarray,
	frame_time: float,
	kernel: np.ndarray,
	morph_operation: str,
	morph_iterations: int,
	min_area: float,
	min_radius: float,
	max_radius: float,
	smoothing_factor: float,
	auto_radius: int,
	auto_alpha: float,
	trail_fade: bool,
	predict_seconds: float,
	velocity_scale: float,
) -> Optional[np.ndarray]:
	target_mask = cv2.inRange(hsv_frame, target.lower.astype(np.uint8), target.upper.astype(np.uint8))
	if morph_iterations > 0:
		if morph_operation == "open":
			target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
			target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
		else:
			target_mask = cv2.erode(target_mask, None, iterations=morph_iterations)
			target_mask = cv2.dilate(target_mask, None, iterations=morph_iterations)

	cnts = cv2.findContours(target_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	previous_point = target.pts[0] if len(target.pts) > 0 and target.pts[0] is not None else None
	best_contour = None
	best_center = None
	best_radius = 0.0
	best_circle_center = None
	best_score = None

	for contour in cnts:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue
		(x_candidate, y_candidate), radius_candidate = cv2.minEnclosingCircle(contour)
		if radius_candidate < min_radius:
			continue
		if max_radius > 0.0 and radius_candidate > max_radius:
			continue
		moments = cv2.moments(contour)
		if moments["m00"] == 0:
			continue
		center_candidate = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
		if previous_point is not None:
			dist = np.linalg.norm(np.array(center_candidate) - np.array(previous_point))
			score = -dist
		else:
			score = area
		if best_contour is None or score > best_score:
			best_contour = contour
			best_center = center_candidate
			best_radius = radius_candidate
			best_score = score
			best_circle_center = (int(x_candidate), int(y_candidate))

	tracked_center: Optional[Tuple[int, int]] = None
	if best_contour is not None and best_center is not None:
		if auto_radius > 0:
			x0 = max(best_center[0] - auto_radius, 0)
			y0 = max(best_center[1] - auto_radius, 0)
			x1 = min(best_center[0] + auto_radius, hsv_frame.shape[1])
			y1 = min(best_center[1] + auto_radius, hsv_frame.shape[0])
			roi = hsv_frame[y0:y1, x0:x1]
			if roi.size > 0:
				roi_flat = roi.reshape(-1, 3)
				lower_sample = np.percentile(roi_flat, 10, axis=0)
				upper_sample = np.percentile(roi_flat, 90, axis=0)
				target.lower = (1.0 - auto_alpha) * target.lower + auto_alpha * lower_sample
				target.upper = (1.0 - auto_alpha) * target.upper + auto_alpha * upper_sample
				target.lower, target.upper = sanitize_bounds(target.lower, target.upper)

		if smoothing_factor > 0:
			current_center = np.array(best_center, dtype=np.float32)
			if target.smoothed_center is None:
				target.smoothed_center = current_center
			else:
				target.smoothed_center = (
					smoothing_factor * current_center + (1.0 - smoothing_factor) * target.smoothed_center
				)
			tracked_center = (
				int(target.smoothed_center[0]),
				int(target.smoothed_center[1]),
			)
		else:
			tracked_center = best_center
		target.last_radius = float(best_radius)
	else:
		target.smoothed_center = None if smoothing_factor > 0 else target.smoothed_center
		target.last_radius = 0.0

	target.pts.appendleft(tracked_center)
	target.timestamps.appendleft(frame_time if tracked_center is not None else None)
	target.last_center = tracked_center
	target.last_velocity = None
	target.last_speed = 0.0

	if tracked_center is not None:
		for j in range(1, len(target.pts)):
			if target.pts[j] is None or target.timestamps[j] is None:
				continue
			delta_t = target.timestamps[0] - target.timestamps[j]
			if delta_t <= 1e-6:
				continue
			displacement = np.array(target.pts[0], dtype=np.float32) - np.array(target.pts[j], dtype=np.float32)
			velocity = displacement / delta_t
			target.last_velocity = velocity
			target.last_speed = float(np.linalg.norm(velocity))
			break

	if target.last_velocity is not None and tracked_center is not None and predict_seconds > 0.0:
		prediction = np.array(tracked_center, dtype=np.float32) + target.last_velocity * predict_seconds
		target.predicted_point = (int(prediction[0]), int(prediction[1]))
	else:
		target.predicted_point = None

	draw_color = tuple(int(np.clip(c, 0, 255)) for c in target.trail_color)
	if best_circle_center is not None and best_radius > 0:
		cv2.circle(bgr_frame, (int(best_circle_center[0]), int(best_circle_center[1])), int(best_radius), (0, 255, 255), 2)
	if tracked_center is not None:
		cv2.circle(bgr_frame, tracked_center, 5, draw_color, -1)

	valid_length = max(1.0, len(target.pts) - 1)
	for idx in range(1, len(target.pts)):
		if target.pts[idx - 1] is None or target.pts[idx] is None:
			continue
		thickness = max(1, int(np.sqrt(target.buffer_len / float(idx + 1)) * 2.5))
		alpha = 1.0
		if trail_fade:
			alpha = max(0.15, 1.0 - (idx / valid_length))
		line_color = tuple(int(np.clip(channel * alpha, 0, 255)) for channel in target.trail_color)
		cv2.line(bgr_frame, target.pts[idx - 1], target.pts[idx], line_color, thickness)

	if target.last_velocity is not None and tracked_center is not None and velocity_scale > 0.0:
		arrow_delta = target.last_velocity * velocity_scale
		arrow_tip = (
			int(tracked_center[0] + arrow_delta[0]),
			int(tracked_center[1] + arrow_delta[1]),
		)
		cv2.arrowedLine(bgr_frame, tracked_center, arrow_tip, draw_color, 2, tipLength=0.3)

	if target.predicted_point is not None:
		x_pred, y_pred = target.predicted_point
		if 0 <= x_pred < bgr_frame.shape[1] and 0 <= y_pred < bgr_frame.shape[0]:
			pred_radius = max(6, int(max(target.last_radius * 0.6, 6)))
			cv2.circle(bgr_frame, target.predicted_point, pred_radius, (255, 255, 0), 2)
			cv2.circle(bgr_frame, target.predicted_point, 2, (255, 255, 0), -1)
		else:
			target.predicted_point = None

	return target_mask

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
ap.add_argument("-p", "--picamera", type=int, choices=[-1, 0, 1], default=-1,
	help="use legacy PiCamera (1=yes,0=no,-1=auto)")
ap.add_argument("-P", "--picamera2", type=int, choices=[-1, 0, 1], default=-1,
	help="use Picamera2/libcamera (1=yes,0=no,-1=auto)")
ap.add_argument("--hsv-lower", type=str_to_hsv,
	help="lower HSV bounds as H,S,V (defaults to 29,86,6)")
ap.add_argument("--hsv-upper", type=str_to_hsv,
	help="upper HSV bounds as H,S,V (defaults to 64,255,255)")
ap.add_argument("--min-radius", type=float, default=3.0,
	help="minimum detected radius (in pixels) to treat as the ball")
ap.add_argument("--max-radius", type=float, default=0.0,
	help="optional maximum radius (pixels) to reject larger objects (0 disables)")
ap.add_argument("--min-area", type=float, default=30.0,
	help="minimum contour area to consider before radius filtering")
ap.add_argument("--resize-width", type=int, default=800,
	help="resize width for processing (<=0 keeps native resolution)")
ap.add_argument("--smoothing", type=float, default=0.2,
	help="exponential smoothing factor for the tracked center (0 disables)")
ap.add_argument("--display-mask", action="store_true",
	help="display the binary mask window alongside the frame")
ap.add_argument("--headless", action="store_true",
	help="skip GUI windows (useful for SSH/headless runs)")
ap.add_argument("--tune", action="store_true",
	help="enable interactive HSV tuning using trackbars (requires GUI)")
ap.add_argument("--auto-range", type=int, default=0,
	help="enable adaptive HSV range updates using a ROI radius in pixels (0 disables)")
ap.add_argument("--auto-alpha", type=float, default=0.25,
	help="smoothing factor (0-1) for adaptive HSV updates")
ap.add_argument("--fps", action="store_true",
	help="log frames-per-second information to stdout")
ap.add_argument("--morph-operation", choices=["erode", "open"], default="erode",
	help="type of morphological cleanup applied to the mask")
ap.add_argument("--morph-iterations", type=int, default=2,
	help="number of iterations for morphological cleanup (0 disables)")
ap.add_argument("--trail-fade", action="store_true",
	help="draw the historical trail with a fading effect")
ap.add_argument("--trail-color", type=str_to_bgr, default=None,
	help="B,G,R color for the trail (defaults to 0,0,255)")
ap.add_argument("--predict-seconds", type=float, default=0.0,
	help="project ball location this many seconds ahead (0 disables)")
ap.add_argument("--velocity-scale", type=float, default=0.1,
	help="scale factor (seconds) used to draw the velocity arrow length")
ap.add_argument("--hud", action="store_true",
	help="overlay HUD with speed, radius, and position information")
ap.add_argument("--ui", action="store_true",
	help="enable the on-screen control panel")
ap.add_argument("--ui-theme", default="DarkBlue3",
	help="theme name for the control panel (DarkBlue3, Dark, Light)")
args = vars(ap.parse_args())

if args["headless"] and (args["display_mask"] or args["tune"]):
	raise ValueError("Headless mode cannot be combined with GUI-related options")
if args["tune"] and args["auto_range"] > 0:
	raise ValueError("Interactive tuning cannot be combined with auto-range updates")
use_ui = bool(args["ui"])
if use_ui and args["headless"]:
	raise ValueError("The control panel requires GUI mode (disable --headless)")
if use_ui and args["tune"]:
	raise ValueError("Trackbar tuning (--tune) cannot be combined with the control panel")

if args["picamera2"] == -1 and args["picamera"] == -1 and Picamera2 is not None and not args.get("video", False):
	args["picamera2"] = 1
	args["picamera"] = 0
if args["picamera2"] == -1:
	args["picamera2"] = 0
if args["picamera"] == -1:
	args["picamera"] = 0

smoothing_factor = max(0.0, min(args["smoothing"], 0.95))
min_radius = max(0.0, args["min_radius"])
max_radius = max(0.0, args["max_radius"])
auto_radius = max(0, args["auto_range"])
auto_alpha = max(0.0, min(args["auto_alpha"], 1.0))
morph_iterations = max(0, args["morph_iterations"])
morph_operation = args["morph_operation"]
min_area = max(0.0, args["min_area"])
resize_width = args["resize_width"]
predict_seconds = max(0.0, args["predict_seconds"])
velocity_scale = max(0.0, args["velocity_scale"])
trail_color_value = args["trail_color"] if args["trail_color"] is not None else (0, 0, 255)
trail_color_default = brighten_color(np.array(trail_color_value, dtype=np.float32), alpha=1.15, beta=15.0)

if max_radius > 0.0 and max_radius <= min_radius:
	raise ValueError("max-radius must be greater than min-radius (or zero to disable)")

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
default_lower = np.array((29, 86, 6), dtype=np.float32)
default_upper = np.array((64, 255, 255), dtype=np.float32)
base_lower = np.array(
	args["hsv_lower"] if args["hsv_lower"] is not None else default_lower,
	dtype=np.float32,
)
base_upper = np.array(
	args["hsv_upper"] if args["hsv_upper"] is not None else default_upper,
	dtype=np.float32,
)
default_name = "Green" if args["hsv_lower"] is None and args["hsv_upper"] is None else "Custom"
targets: List[ColorTarget] = [
	ColorTarget(
		name=default_name,
		lower=base_lower,
		upper=base_upper,
		trail_color=trail_color_default,
		buffer_len=args["buffer"],
	)
]
vs = None

# initialize the video source depending on the arguments supplied
picam2 = None
if args["picamera2"] > 0:
	if Picamera2 is None:
		raise ImportError("Picamera2 is not available. Install it with 'sudo apt install python3-picamera2'.")
	if args.get("video", False):
		raise ValueError("Picamera2 cannot be used together with a video file input.")
	picam2 = Picamera2()
	config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
	picam2.configure(config)
	picam2.start()
elif not args.get("video", False):
	if args["picamera"] > 0:
		vs = VideoStream(usePiCamera=True).start()
	else:
		vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
frame_counter = 0
fps_checkpoint = time.time()
primary_target = targets[0] if targets else None

if args["tune"]:
	if primary_target is None:
		raise ValueError("Interactive tuning requires at least one target")
	cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("HSV Controls", 420, 320)
	for idx, channel in enumerate(("H", "S", "V")):
		cv2.createTrackbar(
			f"Lower {channel}", "HSV Controls", int(primary_target.lower[idx]), 255, lambda _ : None
		)
		cv2.createTrackbar(
			f"Upper {channel}", "HSV Controls", int(primary_target.upper[idx]), 255, lambda _ : None
		)

if args["display_mask"]:
	cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

click_state = {"coords": None}
latest_frame: Optional[np.ndarray] = None
latest_hsv: Optional[np.ndarray] = None
awaiting_pick = False
status_message = "Control panel ready" if use_ui else ""
selected_target_idx = 0 if targets else None
panel = ControlPanel(args["ui_theme"]) if use_ui else None

if not args["headless"]:
	cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

	def handle_mouse(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			click_state["coords"] = (x, y)

	cv2.setMouseCallback("Frame", handle_mouse)

while True:
	if panel:
		if not panel.poll():
			panel = None
			use_ui = False
			awaiting_pick = False
		else:
			for event, payload in panel.pop_events():
				if event == "toggle_target":
					idx = payload if payload is not None else selected_target_idx
					if idx is not None and 0 <= idx < len(targets):
						selected_target_idx = idx
						targets[idx].active = not targets[idx].active
						status_message = f"{targets[idx].name} {'enabled' if targets[idx].active else 'disabled'}"
				elif event == "pick":
					awaiting_pick = True
					status_message = "Click the ball in the video window"
				elif event == "clear":
					if selected_target_idx is not None and 0 <= selected_target_idx < len(targets):
						reset_target_history(targets[selected_target_idx])
						status_message = f"{targets[selected_target_idx].name} trail cleared"
					else:
						for target in targets:
							reset_target_history(target)
						status_message = "All trails cleared"
				elif event == "remove":
					if selected_target_idx is None or selected_target_idx >= len(targets):
						status_message = "Select a ball first"
					elif len(targets) == 1:
						status_message = "Cannot remove the last ball"
					else:
						removed = targets.pop(selected_target_idx)
						status_message = f"Removed {removed.name}"
						selected_target_idx = min(selected_target_idx, len(targets) - 1) if targets else None
						primary_target = targets[0] if targets else None
				elif event == "close":
					panel.destroy()
					panel = None
					use_ui = False
					awaiting_pick = False
					status_message = "Control panel closed"
					break
		if panel:
			panel.refresh_targets(targets, selected_target_idx)
			panel.set_pick_mode(awaiting_pick)
			display_status = "Click the ball in the video window" if awaiting_pick else (status_message or "Use the controls to manage tracking")
			panel.set_status(display_status)

	if picam2 is not None:
		frame = picam2.capture_array()
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	else:
		frame = vs.read()
		frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		break

	if resize_width > 0:
		frame = imutils.resize(frame, width=resize_width)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	latest_frame = frame.copy()
	latest_hsv = hsv.copy()

	if args["tune"] and primary_target is not None:
		lower_vals = np.array([cv2.getTrackbarPos(f"Lower {channel}", "HSV Controls") for channel in ("H", "S", "V")], dtype=np.float32)
		upper_vals = np.array([cv2.getTrackbarPos(f"Upper {channel}", "HSV Controls") for channel in ("H", "S", "V")], dtype=np.float32)
		primary_target.lower, primary_target.upper = sanitize_bounds(lower_vals, upper_vals)

	if awaiting_pick and click_state["coords"] is not None and latest_hsv is not None and latest_frame is not None:
		pick_point = click_state["coords"]
		click_state["coords"] = None
		sample = sample_color_from_point(latest_hsv, latest_frame, pick_point)
		if sample is None:
			awaiting_pick = False
			status_message = "Unable to sample color, try again"
		else:
			lower_sample, upper_sample, mean_bgr = sample
			new_trail_color = brighten_color(mean_bgr, alpha=1.25, beta=20.0)
			default_name = next_target_name([t.name for t in targets], base="Ball")
			target_name = default_name
			cancelled = False
			if panel and panel.is_open():
				try:
					panel.root.lift()
					panel.root.attributes("-topmost", True)
				except tk.TclError:
					pass
				response = simpledialog.askstring(
					"New Ball",
					"Name this ball (leave blank for auto)",
					initialvalue=default_name,
					parent=panel.root,
				)
				try:
					panel.root.attributes("-topmost", False)
				except tk.TclError:
					pass
				if response is None:
					cancelled = True
					status_message = "Color capture cancelled"
				else:
					target_name = response.strip() or default_name
			if not cancelled:
				new_target = ColorTarget(
					name=target_name,
					lower=lower_sample,
					upper=upper_sample,
					trail_color=new_trail_color,
					buffer_len=args["buffer"],
				)
				targets.append(new_target)
				primary_target = targets[0]
				selected_target_idx = len(targets) - 1
				status_message = f"Added {target_name}"
				if panel and panel.is_open():
					panel.refresh_targets(targets, selected_target_idx)
			awaiting_pick = False
			if panel and panel.is_open():
				panel.set_pick_mode(False)

	frame_time = time.time()
	mask_display = None

	for idx, target in enumerate(targets):
		target.set_buffer_len(args["buffer"])
		if not target.active:
			target.smoothed_center = None
			target.last_center = None
			target.last_velocity = None
			target.last_speed = 0.0
			target.predicted_point = None
			continue
		mask_result = process_target(
			target,
			hsv,
			frame,
			frame_time,
			kernel,
			morph_operation,
			morph_iterations,
			min_area,
			min_radius,
			max_radius,
			smoothing_factor,
			auto_radius,
			auto_alpha,
			args["trail_fade"],
			predict_seconds,
			velocity_scale,
		)
		if mask_result is not None:
			mask_display = mask_result if mask_display is None else cv2.bitwise_or(mask_display, mask_result)

	if awaiting_pick:
		cv2.putText(
			frame,
			"Click the ball to capture its color",
			(10, frame.shape[0] - 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 255),
			2,
			cv2.LINE_AA,
		)

	if args["hud"]:
		hud_lines = []
		for target in targets:
			if not target.active:
				continue
			speed_text = f"{target.last_speed:.1f} px/s" if target.last_velocity is not None else "-- px/s"
			radius_text = f"{target.last_radius:.1f} px" if target.last_radius > 0 else "-- px"
			center_text = f"{target.last_center[0]:d},{target.last_center[1]:d}" if target.last_center is not None else "--,--"
			line = f"{target.name}: speed {speed_text} | radius {radius_text} | center {center_text}"
			if target.predicted_point is not None and 0 <= target.predicted_point[0] < frame.shape[1] and 0 <= target.predicted_point[1] < frame.shape[0]:
				line += f" | predicted {target.predicted_point[0]:d},{target.predicted_point[1]:d}"
			hud_lines.append(line)
		y_offset = 25
		for idx, line in enumerate(hud_lines):
			cv2.putText(
				frame,
				line,
				(10, y_offset + idx * 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)

	frame_counter += 1
	if args["fps"] and frame_counter % 30 == 0:
		now = time.time()
		elapsed = max(now - fps_checkpoint, 1e-6)
		fps = 30.0 / elapsed
		print(f"[INFO] approx FPS: {fps:.2f}")
		fps_checkpoint = now

	if not args["headless"]:
		cv2.imshow("Frame", frame)
		if args["display_mask"]:
			if mask_display is None:
				mask_display = np.zeros(frame.shape[:2], dtype=np.uint8)
			cv2.imshow("Mask", mask_display)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

if panel:
	panel.destroy()

# if we are not using a video file, stop/close the camera stream
if picam2 is not None:
	picam2.stop()
elif not args.get("video", False):
	if vs is not None:
		vs.stop()
elif vs is not None:
	vs.release()

# close all windows
cv2.destroyAllWindows()
