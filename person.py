from ultralytics import YOLO
import cv2

# Load model YOLOv8n
model = YOLO(r"D:\practice\test_practice\yolov8n.pt")

# Đường dẫn ảnh cần dự đoán
image_path = r"D:\practice\test_practice\image\4.jpg"  # thay bằng đường dẫn ảnh thật

# Dự đoán
results = model(image_path)

# Lấy đối tượng kết quả đầu tiên
boxes = results[0].boxes
names = model.names

# Đọc ảnh để vẽ
img = cv2.imread(image_path)

# Duyệt qua các box
for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = names[cls_id]
    
    if label == "person":  # Lọc chỉ người
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Hiển thị ảnh kết quả
import matplotlib.pyplot as plt

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Detected People")
plt.axis("off")
plt.show()
