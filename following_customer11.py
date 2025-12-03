import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions
from ultralytics import YOLO

import numpy as np
import cv2
import threading
import math


class DepthToMap(Node):
    def __init__(self):
        super().__init__('depth_to_map_node')

        self.bridge = CvBridge()
        self.K = None
        self.lock = threading.Lock()

        # ▶ 센서용 QoS (가장 최신 프레임만 유지)
        self.qos_sensor = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'

        # ▶ YOLO 로드
        self.get_logger().info("Loading YOLO model...")
        self.yolo = YOLO('/home/rokey/Picker_project/yolo_mixed.pt')
        self.get_logger().info("YOLO loaded.")

        # ▶ 추적할 클래스 이름
        self.target_class = "customer_b"

        self.depth_image = None
        self.rgb_image = None
        self.yolo_running = False

        # ▶ rqt용 이미지 QoS
        self.qos_image = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # ▶ YOLO가 그린 이미지를 퍼블리시할 토픽
        self.yolo_image_pub = self.create_publisher(
            Image,
            'image_yolo',
            self.qos_image
        )

        # ▶ cmd_vel 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'{ns}/cmd_vel',
            10
        )

        # ▶ 추적/탐색 파라미터
        self.follow_distance = 1.1
        self.k_v = 0.8
        self.k_w = 1.2
        self.max_linear_speed = 0.25
        self.max_angular_speed = 0.5

        # 🔹 도리도리 방지용 데드존
        self.dist_deadband = 0.05   # m, 목표 거리 ±10cm 이내면 전진/후진 안 함
        self.angle_deadband = 0.17  # 정규화된 에러(화면 절반 기준 17%) 이하면 회전 안 함

        self.lost_timeout = 1.0
        self.search_angular_speed = 0.5
        self.search_duration = 2 * math.pi / abs(self.search_angular_speed)

        self.state = "IDLE"
        self.last_detection_time = None
        self.search_start_time = None

        # TurtleBot4 네비게이터
        self.navigator = TurtleBot4Navigator()

        if not self.navigator.getDockedStatus():
            self.get_logger().info('Docking before initializing pose')
            self.navigator.dock()

        initial_pose = self.navigator.getPoseStamped(
            [-3.95146, 3.98198],
            TurtleBot4Directions.NORTH
        )
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.navigator.undock()

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        # ▶ 서브스크립션
        self.create_subscription(
            CameraInfo, self.info_topic,
            self.camera_info_callback, self.qos_sensor
        )
        self.create_subscription(
            Image, self.depth_topic,
            self.depth_callback, self.qos_sensor
        )
        self.create_subscription(
            CompressedImage, self.rgb_topic,
            self.rgb_callback, self.qos_sensor
        )

        # ▶ FPS 계산용 변수
        self.rgb_count = 0
        self.depth_count = 0

        # 1초마다 FPS 출력 타이머
        self.fps_timer = self.create_timer(1.0, self.print_fps)


        self.get_logger().info("TF Tree 안정화 시작. 5초 후 변환 시작합니다.")
        self.start_timer = self.create_timer(5.0, self.start_transform)

    def start_transform(self):
        self.get_logger().info("TF Tree 안정화 완료. 변환 + 추적 시작합니다.")
        self.timer = self.create_timer(0.2, self.process_frame)
        self.start_timer.cancel()

    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f"Camera intrinsics received: "
                    f"fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                    f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )
                self.logged_intrinsics = True

    def depth_callback(self, msg):
        # (옵션) 시간 체크
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = now_sec - msg_sec
        if dt > 0.5:
            self.get_logger().warn(
                f"Depth frame too old ({dt:.2f}s delay). Dropping frame."
            )
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # self.depth_count += 1  # FPS 측정 중이면

            if depth is None or depth.size == 0:
                self.get_logger().error("Depth image is empty")
            else:
                if not self.logged_depth_shape:
                    self.get_logger().info(f"Depth image received: {depth.shape}")
                    self.logged_depth_shape = True

            with self.lock:
                self.depth_image = depth

        except Exception as e:
            self.get_logger().error(f"Depth CV bridge conversion failed: {e}")



    def rgb_callback(self, msg):
        # 🔹 먼저 지연시간(dt) 계산
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = now_sec - msg_sec

        # 너무 오래된 프레임이면 버리기 (예: 0.5초 이상)
        if dt > 0.5:
            self.get_logger().warn(
                f"RGB frame too old ({dt:.2f}s delay). Dropping frame."
            )
            return

        # 🔹 실제 디코딩은 여기서 try/except로 감싸기
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # FPS 카운트 쓰고 있으면 여기서 += 1
            # self.rgb_count += 1

            if rgb is None or rgb.size == 0:
                self.get_logger().error("Decoded RGB image is empty")
            else:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f"RGB image decoded: {rgb.shape}")
                    self.logged_rgb_shape = True

            with self.lock:
                self.rgb_image = rgb

        except Exception as e:
            self.get_logger().error(f"Compressed RGB decode failed: {e}")


    def print_fps(self):
        self.get_logger().info(
            f"RGB FPS: {self.rgb_count}   |   Depth FPS: {self.depth_count}"
        )
        self.rgb_count = 0
        self.depth_count = 0



    def process_frame(self):
        if self.yolo_running:
            return

        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None

        if rgb is None:
            return

        self.yolo_running = True
        now = self.get_clock().now()

        try:
            rgb_display = rgb.copy()
            boxes = self.run_yolo(rgb_display)

            target_found = False
            target_cx = None
            target_cy = None
            target_dist = None

            # 🔹 confidence 기준값
                        # 🔹 confidence 기준값
            MIN_CONF = 0.9

            best_box = None
            best_conf = 0.0  # 최고 conf 찾기용

            # 1) target_class 중에서 conf 가장 높은 박스 찾기
            for (x1, y1, x2, y2, name, conf) in boxes:
                if name == self.target_class and conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2, name, conf)

            # 2) best_box만 시각화 (원하면 MIN_CONF 조건도 같이)
            if best_box is not None and best_conf >= MIN_CONF:
                x1, y1, x2, y2, name, conf = best_box
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    rgb_display,
                    f"{name} {conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


            # rqt용 YOLO 이미지 퍼블리시
            img_msg = self.bridge.cv2_to_imgmsg(rgb_display, encoding='bgr8')
            img_msg.header.stamp = now.to_msg()
            img_msg.header.frame_id = 'oakd_rgb_frame'
            self.yolo_image_pub.publish(img_msg)

            # 🔹 car 거리 계산 (conf ≥ 0.9 인 경우에만 사용)
            if best_box is not None and best_conf >= MIN_CONF and depth is not None:
                x1, y1, x2, y2, name, conf = best_box
                cx = int((x1 + x2) / 2)
                # cy = int((y1 + y2) / 2)
                cy = int(y2 - (y2 - y1) * 0.05)


                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    z = float(depth[cy, cx]) / 1000.0
                    if 0.2 < z < 5.0:
                        target_found = True
                        target_cx = cx
                        target_cy = cy
                        target_dist = z
            else:
                # best_box는 있었는데 conf가 너무 낮을 때 로그 찍고 무시하고 싶으면:
                if best_box is not None and best_conf < MIN_CONF:
                    self.get_logger().debug(
                        f"Detected '{self.target_class}' but conf={best_conf:.2f} < {MIN_CONF}"
                    )

            if target_found:
                self.last_detection_time = now
                self.state = "TRACKING"
                self.search_start_time = None
                self.track_target(target_cx, target_cy, target_dist, rgb.shape)
            else:
                if self.last_detection_time is None:
                    self.state = "IDLE"
                    self.stop_robot()
                else:
                    elapsed = (now - self.last_detection_time).nanoseconds * 1e-9
                    if elapsed < self.lost_timeout:
                        self.state = "TRACKING"
                        self.stop_robot()
                    else:
                        if self.state != "SEARCHING":
                            self.state = "SEARCHING"
                            self.search_start_time = now
                            self.get_logger().info(
                                f"Target '{self.target_class}' lost. Start searching (rotate 360 deg)."
                            )
                        self.search_for_target(now)
        except Exception as e:
            self.get_logger().warn(f"Frame processing (YOLO/Publish/Control) error: {e}")
        finally:
            self.yolo_running = False


    def track_target(self, cx, cy, dist, image_shape):
        """car를 일정 거리 두고 따라가는 제어."""
        height, width, _ = image_shape

        # 이미지 중심 대비 x 방향 오차
        center_x = width / 2.0
        error_x = (cx - center_x) / center_x  # -1 ~ +1 정도 (정규화된 값)

        # 거리 오차 (앞/뒤)
        dist_error = dist - self.follow_distance  # 멀면 +, 가까우면 -

        # 🔹 거리 데드존: 목표 거리 ± dist_deadband 이내면 전진 안 함
        if abs(dist_error) < self.dist_deadband:
            dist_error = 0.0

        # 🔹 각도 데드존: 거의 중앙(±angle_deadband)이면 회전 안 함
        if abs(error_x) < self.angle_deadband:
            error_x = 0.0

        # 선속도
        linear_x = self.k_v * dist_error
        # 너무 가까운데 dist_error<0 라고 해서 뒤로 가지 않게 (원하면 뒤로도 가게 풀어도 됨)
        if dist < self.follow_distance and dist_error <= 0:
            linear_x = 0.0

        # 각속도 (좌우 정렬)
        angular_z = - self.k_w * error_x

        # saturate
        linear_x = max(min(linear_x, self.max_linear_speed), -self.max_linear_speed)
        angular_z = max(min(angular_z, self.max_angular_speed), -self.max_angular_speed)

        # 🔹 거리·각도 둘 다 거의 맞으면 완전 정지 (LOG 찍어봐도 좋음)
        if linear_x == 0.0 and angular_z == 0.0:
            # self.get_logger().info("Target aligned & within distance. Holding pose.")
            self.stop_robot()
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)


    def search_for_target(self, now):
        if self.search_start_time is None:
            self.search_start_time = now

        elapsed = (now - self.search_start_time).nanoseconds * 1e-9

        if elapsed < self.search_duration:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = float(self.search_angular_speed)
            self.cmd_vel_pub.publish(twist)
        else:
            self.get_logger().info(
                f"Search finished. Target '{self.target_class}' not found. Go to IDLE."
            )
            self.state = "IDLE"
            self.stop_robot()
            self.last_detection_time = None
            self.search_start_time = None

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def run_yolo(self, rgb_image):
        results = self.yolo(rgb_image)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names[cls_id]
                boxes.append((x1, y1, x2, y2, cls_name, conf))
        return boxes


def main():
    rclpy.init()
    node = DepthToMap()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()