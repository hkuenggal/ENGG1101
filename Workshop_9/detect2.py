# ============================================================
# 🎯 YOLO Object Detection - Test Your Model
# ============================================================
# INSTRUCTIONS:
#   1. Replace MODEL_PATH with the path to your best.pt file
#   2. Replace IMAGE_PATH with the path to your test image
#   3. Run the script and see the results!
#
# HOW TO FIND YOUR FILE PATH:
#   Windows: Right-click the file → "Copy as path" → paste below
#   Mac:     Right-click the file → hold Option → "Copy as Pathname"
#
# EXAMPLE:
#   MODEL_PATH = r"C:\Users\yourname\Downloads\best.pt"
#   IMAGE_PATH = r"C:\Users\yourname\Downloads\test.jpg"
# ============================================================

# 🔧 STEP 1: Change these two lines to your own file paths
MODEL_PATH = r"YOUR_MODEL_PATH\best.pt"
IMAGE_PATH = r"YOUR_IMAGE_PATH\test.jpg"

# ============================================================

from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_PATH, conf=0.25, save=False, verbose=False)

annotated = results[0].plot()
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
Image.fromarray(annotated_rgb).show()

print(f"\n📦 Total detections: {len(results[0].boxes)}\n")
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = model.names[cls]
    print(f"  ✅ Detected: {class_name:<20} | Confidence: {conf:.2f}")
