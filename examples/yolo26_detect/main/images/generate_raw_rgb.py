"""
Generate raw_rgb_bus.h from bus.jpg for bit-exact MCU validation.
Bypasses the ESP32-P4 hardware JPEG decoder so preprocessing
receives pixel-identical input as Python's cv2.imread.

Usage: python generate_raw_rgb.py
Output: raw_rgb_bus.h (in the same directory)
"""
import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load full-size bus.jpg and half-resize (matches notebook preprocessing)
im_full = cv2.imread(os.path.join(SCRIPT_DIR, "bus.jpg"))
h, w = im_full.shape[:2]
im_half = cv2.resize(im_full, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
print(f"Loaded bus.jpg ({w}x{h}) -> half-resized ({w//2}x{h//2})")

# 2. Convert BGR -> RGB (ESP-DL expects RGB888)
im_rgb = cv2.cvtColor(im_half, cv2.COLOR_BGR2RGB)

# 3. Write C header
hw, hh = w // 2, h // 2
flat = im_rgb.flatten().tolist()
out_path = os.path.join(SCRIPT_DIR, "raw_rgb_bus.h")
with open(out_path, "w") as f:
    f.write(f"// Auto-generated from bus.jpg ({hw}x{hh} RGB888)\n")
    f.write(f"// Total bytes: {len(flat)}\n\n")
    f.write(f"const int raw_rgb_bus_width  = {hw};\n")
    f.write(f"const int raw_rgb_bus_height = {hh};\n")
    f.write(f"const int raw_rgb_bus_len    = {len(flat)};\n\n")
    f.write("const uint8_t raw_rgb_bus[] = {\n")
    for i in range(0, len(flat), 16):
        f.write("    " + ",".join(str(v) for v in flat[i:i + 16]))
        if i + 16 < len(flat):
            f.write(",\n")
    f.write("\n};\n")

print(f"Written: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
