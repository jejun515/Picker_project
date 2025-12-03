from ultralytics import YOLO
import cv2
import numpy as np

# YOLO 모델 로드
model = YOLO("/home/rokey/box_count/model_pt/final.pt")

# 클래스 이름 (4개 클래스 가정)
# 모델에 등록된 순서대로 수정하면 됨
class_names = ["S_box", "M_box", "L_box", "XL_box"]

# 이미지 불러오기
img_path = "/home/rokey/box_count/multi_test.png"
img = cv2.imread(img_path)

# **이미지 크기를 640×640으로 resize**
img_resized = cv2.resize(img, (1280, 960))

# 모델 추론
results = model(img_resized)[0]

# 결과 박스
boxes = results.boxes

# 클래스별 개수 카운트
class_counts = {cls_name: 0 for cls_name in class_names}

for box in boxes:
    cls_id = int(box.cls[0])  # 클래스 번호
    if cls_id < len(class_names):
        class_name = class_names[cls_id]
        class_counts[class_name] += 1

# 출력
print("\n===== 클래스별 개수 =====")
for cls_name, cnt in class_counts.items():
    print(f"{cls_name}: {cnt} 개")

print(f"\n전체 박스 개수: {len(boxes)} 개")

# 원한다면 시각화도 가능
annotated = results.plot()  # YOLO가 자동으로 그려줌
cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
