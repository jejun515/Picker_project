from ultralytics import YOLO
import cv2

model = YOLO("/home/rokey/box_count/model_pt/box_yolo8n.pt")


img_path = "/home/rokey/box_count/box_training/test/images/S_box_img_0031.jpg"
img = cv2.imread(img_path)

results = model(img)[0]    # 첫 번째 결과

boxes = results.boxes

print(f"전체 박스 개수: {len(boxes)}")
