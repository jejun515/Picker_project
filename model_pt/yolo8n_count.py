import os
import random
import cv2
from ultralytics import YOLO

# 1. 설정 변수
# YOLO 모델 경로
model_path = "/home/rokey/box_count/model_pt/box_yolo8n.pt"

test_image_dir = "/home/rokey/box_count/box_training/test/images/"

window_name = "YOLOv8 Box Detection Result"

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"오류: YOLO 모델 로드에 실패했습니다. 경로를 확인해주세요: {model_path}")
    print(f"에러 메시지: {e}")
    exit()

try:

    all_files = os.listdir(test_image_dir)

    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print(f"오류: '{test_image_dir}' 폴더에 이미지 파일이 없습니다.")
        exit()

    random_image_file = random.choice(image_files)
    random_img_path = os.path.join(test_image_dir, random_image_file)

    print(f"**랜덤으로 선택된 이미지:** {random_image_file}")

    img = cv2.imread(random_img_path)
    if img is None:
        print(f"오류: 이미지 파일을 로드할 수 없습니다. 경로: {random_img_path}")
        exit()

except Exception as e:
    print(f"오류: 이미지 폴더 처리 중 문제가 발생했습니다: {e}")
    exit()

results = model(img, verbose=False)[0]

boxes = results.boxes

print(f"**탐지된 전체 박스 개수:** {len(boxes)}")
annotated_img = results.plot()
cv2.imshow(window_name, annotated_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

print("\n**테스트 완료.**")