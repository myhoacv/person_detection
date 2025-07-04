import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load model YOLOv8n
model = YOLO("yolov8n.pt")

st.title("👤 YOLOv8 Person Detection")

# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Dự đoán: chỉ phát hiện người (class 0)
    results = model(img, classes=[0])
    boxes = results[0].boxes
    names = model.names

    # Vẽ kết quả lên ảnh
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị ảnh kết quả
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Kết quả phát hiện", use_container_width=True)
