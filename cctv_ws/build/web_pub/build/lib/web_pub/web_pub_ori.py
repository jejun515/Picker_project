import time
import math
import os
import shutil
import sys
from ultralytics import YOLO
from pathlib import Path
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Bool

class YOLOWebcamPublisher(Node):
    def __init__(self, model):
        super().__init__('cctvcam_publisher')
        self.model = model
        self.confidences = []
        self.max_object_count = 0
        self.classNames = model.names
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Bool, 'cctvcam_msg', 10)
        self.should_shutdown = False

        self.bool = False
        self.in_roi_since = None  # ROI 안에 들어온 시간 기록용

        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open webcam.")
            raise RuntimeError("Webcam not available")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        if self.should_shutdown:
            return

        ret, img = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from webcam.")
            return

        h, w, _ = img.shape
        results = self.model(img, stream=True)
        object_count = 0
        fontScale = 1
        yolo_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label = self.classNames.get(cls, f"class_{cls}")
                org = [x1, y1]

                yolo_boxes.append((x1, y1, x2, y2, label, confidence))
                object_count += 1

        # 1) ROI 정의 (화면 가운데 큰 박스)
        roi_x1 = int(w * 0.25)
        roi_y1 = int(h * 0.25)
        roi_x2 = int(w * 0.75)
        roi_y2 = int(h * 0.75)

        cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        cv2.putText(img, "MY ROI", (roi_x1, roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 2) 이번 프레임에서 ROI 안에 들어온 물체가 하나라도 있는지 확인
        any_inside = False

        for (x1, y1, x2, y2, label, confidence) in yolo_boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside_roi = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)
            if inside_roi:
                any_inside = True

            color = (0, 255, 0) if inside_roi else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label}: {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 3) 0.5초 로직 (비블록킹: while 안 돌림)
        now = time.time()
        if any_inside:
            if self.in_roi_since is None:
                self.in_roi_since = now  # 처음 들어온 시점 기록
            # 계속 안에 있고, 3초 이상 지나면 True
            if now - self.in_roi_since >= 0.5:
                self.bool = True
        else:
            # 안에 아무도 없으면 초기화
            self.in_roi_since = None
            self.bool = False

        # 4) 개수 표시 + Bool publish
        self.max_object_count = max(self.max_object_count, object_count)
        cv2.putText(img, f"Objects_count: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 1)

        self.publisher.publish(Bool(data=self.bool))

        # 디버그 겸 화면 보고 싶으면:
        # cv2.imshow("webcam", img)
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.should_shutdown = True


def main():
    model_path = "/home/jejun/Picker_project/training_yolo8/best.pt" #모델 있는 파일 디렉토리로 수정해야함

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        exit(1)

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"❌ Unsupported model format: {suffix}")
        exit(1)

    rclpy.init()
    node = YOLOWebcamPublisher(model)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("🔴 Ctrl+C received. Exiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Shutdown complete.")
        sys.exit(0)

