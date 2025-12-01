import time
import math
import os
import sys
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from sensor_msgs.msg import Image


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

        # --- 웹캠 열기 ---
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open webcam.")
            raise RuntimeError("Webcam not available")

        # 0.1초(10Hz)마다 프레임 처리
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        if self.should_shutdown:
            return

        ret, img = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from webcam.")
            return

        h, w, _ = img.shape

        # --- 평행사변형 ROI 정의 ---
        # 점 순서는 시계 또는 반시계 방향으로 주는 것이 좋음
        roi_points = np.array([
            [25, 125],   # P1
            [600, 100],  # P2
            [640, 340],  # P4
            [0, 340]     # P3
        ], dtype=np.int32)

        # 평행사변형(ROI) 그리기
        cv2.polylines(img, [roi_points], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(img, "MY ROI", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # YOLO 추론
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

                yolo_boxes.append((x1, y1, x2, y2, label, confidence))
                object_count += 1

        # 이번 프레임에서 ROI 안에 들어온 물체가 하나라도 있는지 확인
        any_inside = False

        for (x1, y1, x2, y2, label, confidence) in yolo_boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # --- 평행사변형 ROI 내부 여부 판단 ---
            # pointPolygonTest: >0 inside, 0 on edge, <0 outside
            inside = cv2.pointPolygonTest(roi_points, (cx, cy), False)
            inside_roi = inside >= 0  # 경계 포함해서 ROI로 취급

            if inside_roi:
                any_inside = True

            color = (0, 255, 0) if inside_roi else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label}: {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 0.5초 이상 ROI 안에 있으면 True, 아니면 False
        now = time.time()
        if any_inside:
            if self.in_roi_since is None:
                self.in_roi_since = now  # 처음 들어온 시점 기록
            if now - self.in_roi_since >= 0.5:
                self.bool = True
        else:
            self.in_roi_since = None
            self.bool = False

        # 개수 표시 + 해상도 표시 + Bool publish
        self.max_object_count = max(self.max_object_count, object_count)

        cv2.putText(img, f"Objects_count: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 1)

        cv2.putText(img, f"{w}x{h}", (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.publisher.publish(Bool(data=self.bool))

        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("q pressed, stopping frame processing.")
            self.should_shutdown = True

    def destroy_node(self):
        # 리소스 정리
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    model_path = "/home/jb/Downloads/best.pt"  # 모델 경로

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
