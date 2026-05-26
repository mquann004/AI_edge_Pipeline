from ultralytics import YOLO
import cv2

# Load model YOLO mẫu
model = YOLO("yolov8n.pt")

# Đường dẫn ảnh test
image_path = "images/download.jpg"

# Chạy nhận diện
results = model(image_path)

# Hiển thị kết quả
for result in results:
    annotated_frame = result.plot()

    cv2.imshow("YOLO Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()