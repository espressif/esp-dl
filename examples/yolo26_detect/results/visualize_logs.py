import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import sys

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# Correct path relative to 'results/' folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, "../main/images")
OUTPUT_DIR = CURRENT_DIR

# Model Input Size (Used for rescaling boxes to original image size)
MODEL_WIDTH = 512.0
MODEL_HEIGHT = 512.0

COLORS = ['#FF3838', '#FF9D1C', '#11FF11', '#44BBFF', '#E830E8', '#FFD21F']

# ======================================================================================
# UPDATED LOG DATA 
# ======================================================================================
RAW_LOG_DATA = """
I (2089) image:: bus.jpg
I (4019) yolo26_detect: Pre: 30 ms | Inf: 1770 ms | Post: 20 ms
I (4019) YOLO26: [category: person, score: 0.88, x1: 32, y1: 176, x2: 144, y2: 432]
I (4019) YOLO26: [category: person, score: 0.82, x1: 144, y1: 192, x2: 224, y2: 400]
I (4029) YOLO26: [category: person, score: 0.82, x1: 416, y1: 176, x2: 512, y2: 416]
I (4029) YOLO26: [category: bus, score: 0.73, x1: 16, y1: 112, x2: 512, y2: 368]
I (4039) YOLO26: [category: person, score: 0.50, x1: -8, y1: 264, x2: 40, y2: 408]
I (4049) YOLO26: [category: person, score: 0.27, x1: -8, y1: 264, x2: 40, y2: 408]

I (6039) image:: lego.jpg
I (7959) yolo26_detect: Pre: 20 ms | Inf: 1780 ms | Post: 10 ms
I (7821) YOLO26: [category: 2x2_green, score: 0.56, x1: 228, y1: 136, x2: 328, y2: 300]
I (7821) YOLO26: [category: 2x4_green, score: 0.56, x1: 104, y1: 128, x2: 248, y2: 368]
I (7831) YOLO26: [category: 2x4_green, score: 0.44, x1: 288, y1: 152, x2: 424, y2: 408]
"""

# ======================================================================================
# UPDATED PARSER LOGIC
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
                # Note: Score is now 0.0-1.0 in your new log, convert to 0-100 for display consistency if desired
                score = float(det_match.group(2)) * 100 
                x1 = float(det_match.group(3))
                y1 = float(det_match.group(4))
                x2 = float(det_match.group(5))
                y2 = float(det_match.group(6))
                
                results[current_image].append({
                    "class": cls_name,
                    "score": round(score, 2),
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
            
        print(f"Plotting {img_name}...")
        
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] Could not open image {img_path}: {e}")
            continue

        orig_width, orig_height = img.size
        
        scale_x = orig_width / MODEL_WIDTH
        scale_y = orig_height / MODEL_HEIGHT
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        
        for i, det in enumerate(detections):
            # Rescale
            x1 = det['box'][0] * scale_x
            y1 = det['box'][1] * scale_y
            x2 = det['box'][2] * scale_x
            y2 = det['box'][3] * scale_y
            
            # Matplotlib Rectangle uses (x, y, width, height)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=COLORS[i % len(COLORS)], facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{det['class']} {det['score']}%"
            ax.text(
                x1, y1 - 2, label, 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(facecolor=COLORS[i % len(COLORS)], alpha=0.7, lw=0)
            )
            
        plt.axis('off')
        plt.title(f"Detections: {img_name}")
        
        # Save instead of show
        output_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If args provided, read from file (support piping log output)
        print("Reading logs from CLI argument not implemented in this snippet. Using embedded Lego example.")
    
    data = parse_log_string(RAW_LOG_DATA)
    if data:
        visualize_results(data)
    else:
        print("No data parsed. Check your RAW_LOG_DATA format.")
