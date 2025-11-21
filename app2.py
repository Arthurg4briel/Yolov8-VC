from ultralytics import YOLO
from pathlib import Path
import sys
import cv2 as cv

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'best.pt'

if not MODEL_PATH.exists():
    print(f"Model not found on path: {MODEL_PATH}")
    sys.exit(1)

try:
    model = YOLO(str(MODEL_PATH))
    print(f"Modelo {MODEL_PATH.name} loaded succesfully.")
except Exception as e:
    print(f"Error running YOLO model: {e}")
    sys.exit(1)

CAP = cv.VideoCapture(1)

if not CAP.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)
print("Webcam opened successfully. Press 'q' to quit.")

while CAP.isOpened():
    success, frame = CAP.read()
    if not success:
        print("Error: Could not read frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.5,
        stream=True,
        verbose=False
    )

    detected_frame = next(results).plot()
    cv.imshow("YOLO Live Detection", detected_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv.destroyAllWindows()
print("Webcam and windows closed. Exiting.")