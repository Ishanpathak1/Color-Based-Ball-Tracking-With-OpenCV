# Ball Tracking with Computer Vision

A comprehensive ball tracking system implementing multiple computer vision approaches: traditional color-based detection, modern deep learning (YOLO), and hybrid methods.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Project Overview

This project demonstrates three different approaches to ball tracking, from traditional computer vision to state-of-the-art deep learning:

1. **Color-Based Tracking** - Fast HSV color space detection
2. **Hybrid Tracking** - Combines color-based and YOLO detection
3. **YOLO-Based Tracking** - Deep learning object detection
4. **Custom Trained Model** - YOLOv8 fine-tuned on custom dataset (96.2% accuracy!)

## 📊 Method Comparison

| Method | Speed (FPS) | Accuracy | Lighting Robust | Setup Complexity | Use Case |
|--------|-------------|----------|-----------------|------------------|----------|
| **Color-Based** | 350+ | ~70-80% |  No |  Easy | Fast prototyping, controlled lighting |
| **Hybrid** | 15-30 | ~85-90% |  Partial |  Medium | Balanced speed/accuracy |
| **YOLO (Pre-trained)** | 15-30 | 40-60% |  Yes |  Medium | General objects |
| **YOLO (Custom)** | 8-15 | **96.2%** |  Yes |  Advanced | Production-ready, specific ball |

##  Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_yolo.txt
```

### Running Different Methods

#### 1. Color-Based Tracking (Fastest)

```bash
python ball_tracking.py
```

**Features:**
-  Ultra-fast (350+ FPS)
-  Real-time HSV color picker (GUI)
-  Trajectory tracking with history
-  Works great in controlled lighting

**Controls:**
- Click on ball to auto-detect color
- Adjust HSV sliders for fine-tuning
- Press 'q' to quit

#### 2. YOLO-Based Tracking (Most Accurate)

```bash
# With pre-trained model
python yolo_ball_tracking.py --hud

# With custom trained model (RECOMMENDED!)
python yolo_ball_tracking.py --model trained_models/custom_ball/weights/best.pt --hud
```

**Features:**
-  96.2% accuracy (custom model)
-  Works with any lighting conditions
-  Multi-ball tracking with unique IDs
-  Trajectory prediction
-  Real-time HUD display

**Options:**
```bash
--model PATH          # Path to YOLO model (.pt file)
--video PATH          # Use video file instead of webcam
--conf FLOAT          # Confidence threshold (default: 0.3)
--hud                 # Show performance HUD
--no-display          # Run without GUI (save to file only)
```

#### 3. Advanced YOLO with Kalman Filter

```bash
python yolo_ball_tracking_advanced.py --model trained_models/custom_ball/weights/best.pt --hud
```

**Features:**
-  Same accuracy as YOLO
-  Kalman filter for smooth tracking
-  Better handling of fast motion
-  Motion blur reduction
-  Handles temporary occlusions

#### 4. Hybrid Tracking (Balanced)

```bash
python hybrid_ball_tracking.py --hud
```

**Features:**
-  Combines color-based speed with YOLO accuracy
-  Falls back to color when YOLO fails
-  Best of both worlds
-  Shows both methods' results

#### 5. Compare All Methods

```bash
python compare_methods.py --yolo-model trained_models/custom_ball/weights/best.pt
```

**Features:**
-  Side-by-side comparison
-  Real-time performance metrics
-  Split-screen visualization
-  FPS and detection statistics

##  Project Structure

```
Color-Based-Ball-Tracking-With-OpenCV/
├── ball_tracking.py                    # Color-based tracking (original)
├── yolo_ball_tracking.py               # YOLO-based tracking
├── yolo_ball_tracking_advanced.py      # YOLO + Kalman filter
├── hybrid_ball_tracking.py             # Hybrid approach
├── compare_methods.py                  # Method comparison tool
├── train_custom_yolo.py                # Custom model training script
│
├── trained_models/
│   └── custom_ball/
│       ├── weights/
│       │   ├── best.pt                 # ⭐ Custom trained model (96.2% accuracy)
│       │   └── last.pt                 # Last training checkpoint
│       ├── results.png                 # Training progress graphs
│       ├── confusion_matrix.png        # Model performance matrix
│       └── *.jpg                       # Training visualizations
│
├── final_dataset/
│   ├── images/                         # 1,850 labeled training images
│   ├── labels/                         # YOLO format annotations
│   └── data.yaml                       # Dataset configuration
│
├── requirements.txt                    # Color-based dependencies
├── requirements_yolo.txt               # YOLO dependencies
├── README.md                           # This file
└── TRAINING_GUIDE.md                   # Custom model training guide
```

##  Method Details

### 1. Color-Based Tracking

**How it works:**
```
Image → HSV Color Space → Color Mask → Contour Detection → Ball Position
```

**Advantages:**
-  Extremely fast (350+ FPS)
-  Simple to understand
-  No training required
-  Low computational requirements

**Disadvantages:**
-  Sensitive to lighting changes
-  Requires color tuning per environment
-  Struggles with similar colors in background
-  Poor with motion blur

**Best for:** Prototyping, controlled lighting, real-time applications

---

### 2. YOLO-Based Tracking (Pre-trained)

**How it works:**
```
Image → YOLOv8 Network (80 classes) → Filter "sports ball" → Ball Position
```

**Advantages:**
-  Works with various lighting
-  Handles motion blur better
-  No color calibration needed

**Disadvantages:**
-  Only 40-60% accuracy (not trained on your specific ball)
-  Slower than color-based
-  May detect other round objects

**Best for:** General purpose ball detection

---

### 3. Custom Trained YOLO ( RECOMMENDED)

**How it works:**
```
Image → Custom YOLOv8 (trained on YOUR ball) → High Accuracy Detection
```

**Training Details:**
- Dataset: 1,850 labeled images
- Training time: 16 hours on CPU
- Final accuracy: **96.2% mAP50**
- Model size: 6.2 MB

**Advantages:**
-  **96.2% accuracy** (near-professional!)
-  Trained specifically on YOUR ball
-  Robust to lighting, angles, motion blur
-  Production-ready performance
-  Works in any condition

**Disadvantages:**
-  Requires training data collection
-  Training takes time (one-time cost)
-  Slower inference (8-15 FPS on CPU)

**Best for:** Production applications, professional projects, portfolio

---

### 4. Hybrid Tracking

**How it works:**
```
If YOLO detects ball → Use YOLO detection
Else → Fall back to color-based detection
```

**Advantages:**
-  Combines speed and accuracy
-  More robust than color-alone
-  Good fallback mechanism

**Disadvantages:**
-  Complexity of maintaining two systems
-  Still affected by color-based limitations

**Best for:** Balanced applications where both speed and accuracy matter

---

### 5. Advanced YOLO with Kalman Filter

**How it works:**
```
Image → YOLO Detection → Kalman Filter Prediction → Smooth Tracking
```

**Advantages:**
-  Smoother trajectories
-  Handles temporary occlusions
-  Better motion prediction
-  Reduces jitter in tracking

**Disadvantages:**
-  Slightly more complex
-  Similar speed to regular YOLO

**Best for:** High-quality tracking, sports analysis, trajectory prediction

##  Performance Benchmarks

Tested on Intel Core i7-9750H CPU @ 2.60GHz:

```
Method                    FPS    Detections/Frame   Accuracy   Robustness
────────────────────────────────────────────────────────────────────────
Color-Based               351         0.72          ~75%       Low
Hybrid                     18         0.68          ~85%       Medium
YOLO (Pre-trained)         15         0.65          ~55%       High
YOLO (Custom)              12         0.71          96.2%      Very High
YOLO + Kalman              10         0.71          96.2%      Very High
```

##  Custom Model Training

Want to train your own model? See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for:
- Data collection process (1,850 images)
- Labeling workflow
- Training configuration
- Performance optimization
- Troubleshooting tips

**Quick training:**
```bash
python train_custom_yolo.py --data final_dataset/data.yaml --epochs 50 --device cpu
```

##  Use Cases

### Color-Based Tracking
- Rapid prototyping
- Educational projects
- Controlled laboratory environments
- Games/entertainment with fixed lighting

### Custom YOLO Model
- Professional sports analysis
- Robotic vision systems
- Autonomous vehicles
- Production deployments
- Portfolio/resume projects

### Hybrid Approach
- Mobile applications
- Edge devices with limited compute
- Applications requiring graceful degradation

##  Configuration

### Color-Based Tracking
Edit HSV ranges in `ball_tracking.py` or use GUI sliders:
```python
lower_hsv = [H_min, S_min, V_min]
upper_hsv = [H_max, S_max, V_max]
```

### YOLO Tracking
Adjust confidence threshold:
```bash
python yolo_ball_tracking.py --conf 0.5  # Higher = fewer false positives
```

### Training Parameters
Modify `train_custom_yolo.py`:
```python
--epochs 50         # Number of training epochs
--batch 16          # Batch size
--imgsz 640         # Image size
--device cpu        # cpu or cuda
```

##  Results

### Custom Model Performance

```
Final Metrics (50 epochs, 16 hours training):
├── Precision:    98.5%  (98.5% of detections are correct)
├── Recall:       93.2%  (catches 93.2% of all balls)
├── mAP50:        96.2%  (overall accuracy)
└── mAP50-95:     84.4%  (precision across all thresholds)

Speed:
├── Preprocessing:  2.2 ms/image
├── Inference:      66.5 ms/image
├── Postprocess:    0.7 ms/image
└── Total:          ~14 FPS (CPU), ~60+ FPS (GPU)
```

### Training Progress

Improvement over 50 epochs:
- Epoch 1:  87.1% mAP50
- Epoch 10: 94.1% mAP50
- Epoch 25: 95.8% mAP50
- Epoch 50: 96.2% mAP50 ⭐

##  Troubleshooting

### Color-Based Issues
- **Ball not detected:** Adjust HSV sliders or click on ball to auto-calibrate
- **Too many false positives:** Increase minimum contour area
- **Lighting changes:** Recalibrate colors for new environment

### YOLO Issues
- **Low FPS:** Use GPU (`--device cuda`) or reduce image size
- **No detections:** Lower confidence threshold (`--conf 0.2`)
- **False positives:** Increase confidence threshold (`--conf 0.5`)

### Training Issues
- **Out of memory:** Reduce batch size (`--batch 8`)
- **Slow training:** Use GPU or reduce epochs
- **Low accuracy:** Collect more diverse training data

##  Technologies Used

- **Python 3.12** - Programming language
- **OpenCV 4.8** - Computer vision library
- **Ultralytics YOLOv8** - Object detection framework
- **PyTorch 2.2** - Deep learning backend
- **NumPy** - Numerical computing
- **imutils** - OpenCV helper functions

## Learning Outcomes

This project demonstrates:
- Traditional computer vision (HSV, morphology, contours)
- Deep learning object detection (YOLO architecture)
- Transfer learning and fine-tuning
- Data collection and labeling workflows
- Model training and evaluation
- Performance benchmarking
- Multi-method comparison

##  Future Improvements

- [ ] GPU optimization for real-time performance
- [ ] Multi-camera 3D ball tracking
- [ ] Ball trajectory prediction with physics models
- [ ] Speed and spin estimation
- [ ] Export model to mobile (CoreML, TFLite)
- [ ] Web-based inference (ONNX.js)
- [ ] Real-time streaming to dashboard

##  License

MIT License - See LICENSE file for details

##  Acknowledgments

- **YOLOv8** by Ultralytics
- **OpenCV** community
- Original color-based tracking inspiration from pyimagesearch

##  Contact

For questions or collaboration:
- Project: Color-Based Ball Tracking with OpenCV
- Custom Model: 96.2% accuracy on 1,850 training images

---

** Star this project if you found it helpful!**

** See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions**
