import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# Correct path relative to 'results/' folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, "../main/images")
OUTPUT_DIR = CURRENT_DIR

# Model Input Size
MODEL_WIDTH = 512
MODEL_HEIGHT = 512

COLORS = ['#FF3333', '#3399FF', '#33CC66', '#FF9900', '#CC33FF', '#00CCCC']

# ======================================================================================
# UPDATED LOG DATA 
# ======================================================================================
RAW_LOG_DATA = """
I (2160) image:: bus.jpg
I (4350) yolo26_detect: Pre: 12 ms | Inf: 2067 ms | Post: 13 ms
I (4350) YOLO26: [category: person, score: 0.86, x1: 87, y1: 187, x2: 176, y2: 428]
I (4360) YOLO26: [category: bus, score: 0.83, x1: 69, y1: 109, x2: 447, y2: 353]
I (4360) YOLO26: [category: person, score: 0.81, x1: 169, y1: 194, x2: 229, y2: 406]
I (4370) YOLO26: [category: person, score: 0.77, x1: 380, y1: 187, x2: 449, y2: 416]
I (4380) YOLO26: [category: person, score: 0.53, x1: 63, y1: 263, x2: 95, y2: 414]
I (4380) image:: person.jpg
I (6600) yolo26_detect: Pre: 11 ms | Inf: 2066 ms | Post: 13 ms
I (6600) YOLO26: [category: person, score: 0.81, x1: 330, y1: 171, x2: 405, y2: 378]
I (6600) YOLO26: [category: bicycle, score: 0.71, x1: 188, y1: 307, x2: 388, y2: 409]
I (6610) YOLO26: [category: bicycle, score: 0.44, x1: 121, y1: 131, x2: 193, y2: 182]
I (6610) image:: lego.jpg
I (8810) yolo26_detect: Pre: 15 ms | Inf: 2066 ms | Post: 13 ms
I (8810) YOLO26: [category: remote, score: 0.34, x1: 293, y1: 158, x2: 421, y2: 401]
"""

# ======================================================================================
# ESP-DL BIT-EXACT NEAREST-NEIGHBOR LETTERBOX
# ======================================================================================

def espdl_preprocess(img_bgr: np.ndarray, target_w: int = MODEL_WIDTH, target_h: int = MODEL_HEIGHT) -> np.ndarray:
    """
    Replicates the ESP-DL ImagePreprocessor letterbox exactly:
      - Nearest-neighbor resize preserving aspect ratio
      - Zero (114 gray) padding to fill the target canvas
    Returns a uint8 RGB image of shape (target_h, target_w, 3).
    """
    src_h, src_w = img_bgr.shape[:2]

    # Compute scale (same formula as ESP-DL)
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)   # floor, same as C int cast
    new_h = int(src_h * scale)

    # Nearest-neighbor resize (CV_INTER_NEAREST matches firmware)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center padding
    pad_top    = (target_h - new_h) // 2
    pad_left   = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right  = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)   # ESP-DL uses 114 gray pad
    )

    # BGR → RGB for matplotlib display
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


# ======================================================================================
# PARSER LOGIC
# ======================================================================================

def parse_log_string(log_content):
    results = {}
    current_image = None
    
    # Pattern for "I (1987) image:: bus.jpg"
    img_pattern = re.compile(r"image::\s*(.+)")
    
    # Pattern for "I (5577) YOLO26: [category: person, score: 0.88, x1: 32, y1: 176, x2: 144, y2: 432]"
    det_pattern = re.compile(
        r"category:\s*([^,]+),\s*score:\s*([\d.]+),\s*x1:\s*([-\d.]+),\s*y1:\s*([-\d.]+),\s*x2:\s*([-\d.]+),\s*y2:\s*([-\d.]+)"
    )
    
    for line in log_content.strip().split('\n'):
        line = line.strip()
        
        # 1. Look for a new image filename
        img_match = img_pattern.search(line)
        if img_match:
            current_image = img_match.group(1).strip()
            results[current_image] = []
            continue
            
        # 2. Look for detection data
        if current_image:
            det_match = det_pattern.search(line)
            if det_match:
                cls_name = det_match.group(1)
                score = float(det_match.group(2))
                x1 = int(float(det_match.group(3)))
                y1 = int(float(det_match.group(4)))
                x2 = int(float(det_match.group(5)))
                y2 = int(float(det_match.group(6)))
                
                results[current_image].append({
                    "class": cls_name,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                })
    
    return results

# ======================================================================================
# VISUALIZATION LOGIC
# ======================================================================================

def visualize_results(results):
    for img_name, detections in results.items():
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {img_path}")
            continue
            
        print(f"Plotting {img_name} ({len(detections)} detections)...")
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ERROR] cv2 could not read: {img_path}")
            continue

        # Apply the ESP-DL exact preprocessing letterbox!
        canvas_rgb = espdl_preprocess(img_bgr, MODEL_WIDTH, MODEL_HEIGHT)
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(canvas_rgb)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            score = det["score"]
            cls = det["class"]

            # Clamp to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(MODEL_WIDTH, x2); y2 = min(MODEL_HEIGHT, y2)

            color = COLORS[i % len(COLORS)]
            print(f"  [{i+1}] {cls} {score:.2f}  box=[{x1},{y1},{x2},{y2}]")
            
            # Matplotlib Rectangle uses (x, y, width, height)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{cls} {score:.2f}"
            ax.text(
                x1, max(y1 - 6, 0), label, 
                color='white', fontsize=11, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.7, lw=0, pad=2)
            )
            
        plt.axis('off')
        plt.title(f"YOLO26 Detect: {img_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save instead of show
        output_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If args provided, read from file
        try:
            with open(sys.argv[1], 'r') as f:
                log_data = f.read()
            data = parse_log_string(log_data)
        except Exception as e:
            print(f"Failed to read file: {e}")
            sys.exit(1)
    else:
        # Else use embedded Example
        data = parse_log_string(RAW_LOG_DATA)
        
    if data:
        visualize_results(data)
    else:
        print("No data parsed. Check your RAW_LOG_DATA format or pipe input.")
