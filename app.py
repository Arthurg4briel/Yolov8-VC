from ultralytics import YOLO
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / 'images_test' / 'book1.jpg' 
MODEL_PATH = BASE_DIR / 'model' / 'best.pt' 

# Verifying the files existance
if not MODEL_PATH.exists():
    print(f"Model not found on path: {MODEL_PATH}")
    sys.exit(1)

# Verifying the image existance
if not IMAGE_PATH.exists():
    print(f"Test image not found on path: {IMAGE_PATH}")
    print(f"Verify if image is on: '{BASE_DIR / 'images_test'}'")
    sys.exit(1)

# Running the model
try:
    model = YOLO(str(MODEL_PATH)) 
    print(f"✅ Modelo {MODEL_PATH.name} loaded succesfully.")
except Exception as e:
    print(f"Error running YOLO model: {e}")
    sys.exit(1)

print(f"🔍 Analysing image: {IMAGE_PATH.name}...")

# Executing prediction
results = model.predict(
    source=str(IMAGE_PATH),
    save=True,      
    conf=0.5,       
    name='results_test1' 
)

print("\nDetailed results: ")
for r in results:
    boxes = r.boxes 
    print(f"Total detecctions found: {len(boxes)}")
    for i, box in enumerate(boxes):
        coords = box.xyxy[0].tolist() 
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())

        print(f"  Detecction {i+1}: Class: {class_name}, Confidence: {confidence:.2f}")

print("\nImage with detecctions saved on: runs/detect/")