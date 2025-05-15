from flask import Flask, Response, jsonify
from flask_cors import CORS
from threading import Lock

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from my_models import efficientnet_v2_s
from torchvision import transforms
from my_models import efficientnet_v2_s
import time
import numpy as np

app = Flask(__name__)
CORS(app)

# Shared state for latest prediction
latest_prediction = {"label": "Initializing...", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
prediction_lock = Lock()

# Load model
cam_IP = "rtsp://192.168.60.101:8554/"
model = efficientnet_v2_s(weights=None)
num_classes = 4
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("models/model8_640_368_ER_AUG/model8.pth", map_location="cpu"), strict=True)
print("Model loaded successfully")
model.load_state_dict(torch.load("models/model8_640_368_ER_AUG/model8.pth"))
model.eval()
print("Model loaded successfully")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Resize/mask setup
original_width, original_height = 1280, 720
new_width, new_height = 640, 368
original_roi = [525, 215, 150, 400]
scaled_roi = [
    int(original_roi[0] * (new_width / original_width)),
    int(original_roi[1] * (new_height / original_height)),
    int(original_roi[2] * (new_width / original_width)),
    int(original_roi[3] * (new_height / original_height))
]
mask = cv2.imread("imgs/bin_mask_opt.jpg")
mask_resized = cv2.resize(mask, (scaled_roi[2], scaled_roi[3]))

def resize_to_height(image, height):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(aspect_ratio * height)
    return cv2.resize(image, (new_width, height))

def preprocess_frame(frame):
    A, B = 109, 262
    img_cropped = frame[A:A+204, B:B+75]
    masked_img = cv2.bitwise_and(img_cropped, mask_resized)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    img_cropped_resized = resize_to_height(img_cropped, new_height)
    masked_img_resized = resize_to_height(masked_img, new_height)
    concat_crop_mask = cv2.hconcat([img_cropped_resized, masked_img_resized])

    pil_frame = transforms.functional.to_pil_image(masked_img_rgb)
    transformed_frame = transform(pil_frame)
    return transformed_frame.unsqueeze(0), concat_crop_mask

def generate_frames():
    camera = cv2.VideoCapture(cam_IP, cv2.CAP_FFMPEG)
    if not camera.isOpened():
        print("Could not open stream — fallback to dummy feed.")
        while True:
            blank = np.zeros((368, 640, 3), dtype=np.uint8)
            _, jpeg = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)

    pred_prev = 2  # Assume "No Anomaly" baseline
    counter = 0
    timer = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        input_tensor, concat_crop_mask = preprocess_frame(frame)
        with torch.no_grad():
            prediction = model(input_tensor)

        predicted_class = prediction.argmax(dim=1).item()
        if pred_prev == predicted_class and predicted_class != 2:
            counter += 1
        else:
            counter = 0

        # Label mapping
        classes = ["Fallen After", "Fallen Before", "No Anomaly", "No Lid"]
        Prediction = classes[predicted_class]
        color = (0, 255, 0) if Prediction == "No Anomaly" else (0, 0, 255)

        cv2.putText(frame, f"Prediction: {Prediction}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if counter >= 5 or (0 < timer <= 18):
            cv2.putText(frame, "ERROR!", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            timer += 1

        pred_prev = predicted_class
        if timer == 20:
            counter = 0
            timer = 0

        # Update prediction state for /predict
        if Prediction != "No Anomaly":
            with prediction_lock:
                latest_prediction["label"] = Prediction
                latest_prediction["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        concat_all = cv2.hconcat([frame, concat_crop_mask])
        # Optional: show locally
        # cv2.imshow("Stream", concat_all)

        _, jpeg = cv2.imencode('.jpg', concat_all)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    with prediction_lock:
        return jsonify(
            prediction=latest_prediction["label"],
            timestamp=latest_prediction["timestamp"]
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
