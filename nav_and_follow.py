import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from cv_bridge import CvBridge

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Twist

# =====================================
# tf_transformations 모듈 import 에러 방지
import numpy as np
if not hasattr(np, "float"):
    np.float = float
# =====================================

from tf_transformations import quaternion_from_euler
from ultralytics import YOLO
import cv2
import threading
import math
import os


DETECT_TARGET = [0]  # 웹캠 YOLO에서 사용할 클래스 ID

# 사분면 별 미리 따놓은 좌표 딕셔너리
QUADRANT_TARGET_POSES = {
    1: (-5.69312, 5.59361, math.pi / 2),
    2: (-8.55142, 4.34779, 3 * math.pi / 4),
    3: (-7.87239, 1.90992, -3 * math.pi / 4),
    4: (-4.25377, 2.85634, -math.pi / 2)
}


class NavAndFollow(Node):
    def __init__(self):
        super().__init__('nav_and_follow_node')

        self.get_logger().info('NavAndFollow 노드 시작')

        # phase:
        #   'QUADRANT_NAV' : 웹캠으로 사분면 감지 → Nav2로 goal 전송
        #   'SEARCH_FOLLOW': Nav2 이동 완료 후, OAK-D로 360도 회전 + 추종
        self.phase = 'QUADRANT_NAV'

        self.navigating = False  # 현재 Nav2가 goal 수행 중인지

        # ===================
        # Webcam 객체 인식 모듈
        # ===================
        webcam_model_path = os.path.join("/home/rokey/hj/Picker_project/webcam_final.pt")
        self.webcam_yolo = YOLO(webcam_model_path)

        WEBCAM_PORT = 2 # USB 포트 번호 (불일치시 수정 필요)
        self.cap = cv2.VideoCapture(WEBCAM_PORT)
        if not self.cap.isOpened():
            self.get_logger().error("[ERROR] 웹캠을 열 수 없습니다.")
            raise IOError

        self.bridge = CvBridge()

        # Webcam에서 찍은 사진에서 YOLO detection 수행한 annotation 이미지 (디버깅용, 필수 X)
        self.image_publisher = self.create_publisher(Image, '/webcam/annotated_frame', 10)

        # Nav2 액션 클라이언트 (좌표 이동 명령 전달)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initial pose 설정 및 Nav2 활성화
        self.navigator = TurtleBot4Navigator()
        initial_pose = self.navigator.getPoseStamped(
            [-3.95146, 3.98198],
            TurtleBot4Directions.NORTH
        )
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 활성화 완료")

        self.previous_quadrant = -1
        self.detect_timer = self.create_timer(0.5, self.detection_loop) # 0.5초마다 YOLO detection 수행

        # ================================
        # Depth 카메라 기반 객체 Follower 코드
        # ================================
        self.K = None
        self.lock = threading.Lock()

        # 센서용 QoS
        self.qos_sensor = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # 네임스페이스에 따라 카메라 토픽 결정 (/robot2 등)
        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'

        # OAK-D용 YOLO 모델 (추종용)
        amr_model_path = os.path.join("/home/rokey/hj/Picker_project/amr_final.pt")
        self.follow_yolo = YOLO(amr_model_path)
        self.get_logger().info("AMR YOLO 모델 로드 완료.")

        # 추적할 클래스 이름
        self.target_class = "customer_b"

        self.depth_image = None
        self.rgb_image = None
        self.yolo_running = False

        # rqt용 YOLO 이미지 QoS
        self.qos_image = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # YOLO가 그린 이미지를 퍼블리시할 토픽
        self.yolo_image_pub = self.create_publisher(
            Image,
            'image_yolo',
            self.qos_image
        )

        # cmd_vel 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'{ns}/cmd_vel',
            10
        )

        # 추적/탐색 파라미터
        self.follow_distance = 1.1
        self.k_v = 0.8
        self.k_w = 1.2
        self.max_linear_speed = 0.25
        self.max_angular_speed = 0.5

        self.dist_deadband = 0.05   # m
        self.angle_deadband = 0.17  # 정규화된 x오차

        self.lost_timeout = 1.0
        self.search_angular_speed = 0.5
        self.search_duration = 2 * math.pi / abs(self.search_angular_speed)

        self.state = "IDLE"  # IDLE / TRACKING / SEARCHING
        self.last_detection_time = None
        self.search_start_time = None

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        # OAK-D 구독
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

        # FPS 측정용 (원하면 활성화)
        self.rgb_count = 0
        self.depth_count = 0
        self.fps_timer = self.create_timer(1.0, self.print_fps)

        # OAK-D 처리 타이머 (0.2초마다 실행)
        # 단, phase가 'SEARCH_FOLLOW'일 때만 실제로 추적/회전 동작하도록 process_frame 안에서 체크할 것.
        self.process_timer = self.create_timer(0.2, self.process_frame)

        self.get_logger().info("NavAndFollow 초기화 완료")

    # =========================================================
    # 1) MoveRobot 파트: 웹캠 + Nav2 (사분면 기반 이동)
    # =========================================================
    def get_quadrant(self, xc, yc, frame_cx, frame_cy):
        if (xc >= frame_cx) and (yc <= frame_cy):
            return 1
        elif (xc <= frame_cx) and (yc <= frame_cy):
            return 2
        elif (xc <= frame_cx) and (yc >= frame_cy):
            return 3
        else:
            return 4

    def send_nav2_goal(self, x, y, theta):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('[WARN] Nav2 Action 서버에 접속할 수 없습니다.')
            return

        quats = quaternion_from_euler(0.0, 0.0, theta)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = quats[2]
        goal_msg.pose.pose.orientation.w = quats[3]

        self.get_logger().info(f'[INFO] Nav2 goal 전송: x={x:.2f}, y={y:.2f}, theta={theta:.2f}')
        self.navigating = True
        send_future = self._action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.nav_goal_response_callback)

    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('[WARN] Nav2 goal이 거절되었습니다.')
            self.navigating = False
            return

        self.get_logger().info('[INFO] Nav2 goal이 수락되었습니다.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        result = future.result()
        status = result.status
        self.get_logger().info(f'[INFO] Nav2 결과 status: {status}')
        self.navigating = False

        self.phase = 'SEARCH_FOLLOW'
        self.state = 'SEARCHING'
        self.search_start_time = self.get_clock().now()
        self.last_detection_time = None
        self.get_logger().info('[STATE] phase=SEARCH_FOLLOW, state=SEARCHING (360도 회전 준비)')

    def detection_loop(self):
        # Nav2가 이미 수행중이거나, 이미 추종 모드라면 스킵
        if self.phase != 'QUADRANT_NAV' or self.navigating:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("[WARN] 카메라 프레임을 읽을 수 없습니다.")
            return

        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        results = self.webcam_yolo.predict(frame, classes=DETECT_TARGET, conf=0.5, verbose=False)
        annotated_frame = frame.copy()
        cv2.circle(annotated_frame, (frame_cx, frame_cy), 4, (0, 255, 255), -1)

        new_quadrant = -1

        for r in results:
            if len(r.boxes) > 0:
                box = r.boxes[0]
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]

                xc = int((x1 + x2) / 2)
                yc = int((y1 + y2) / 2)
                new_quadrant = self.get_quadrant(xc, yc, frame_cx, frame_cy)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.circle(annotated_frame, (xc, yc), 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame,
                    f'Q{new_quadrant}',
                    (xc, yc + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                break

        # 사분면 변경 감지 + Nav2 goal 전송
        if new_quadrant != -1 and new_quadrant != self.previous_quadrant:
            self.get_logger().info(
                f"[INFO] 고객 이동 감지: {self.previous_quadrant}사분면 -> {new_quadrant}사분면"
            )
            if new_quadrant in QUADRANT_TARGET_POSES:
                x, y, theta = QUADRANT_TARGET_POSES[new_quadrant]
                self.send_nav2_goal(x, y, theta)
                self.previous_quadrant = new_quadrant

        # Annotated Image를 ROS2 토픽으로 발행
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_publisher.publish(ros_image_msg)
        except Exception as e:
            self.get_logger().error(f'[ERROR] Annotated frame 발행 실패: {e}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shutdown_node()

    # =========================================================
    # 2) DepthToMap 파트: OAK-D + Depth 추종
    # =========================================================
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
        # 시간 체크 (너무 오래된 프레임은 버림)
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
        # 시간 체크
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = now_sec - msg_sec

        if dt > 0.5:
            self.get_logger().warn(
                f"RGB frame too old ({dt:.2f}s delay). Dropping frame."
            )
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
        # FPS 카운터를 쓰고 싶으면 depth_callback / rgb_callback에서 += 1 하면 됨
        self.get_logger().info(
            f"RGB FPS: {self.rgb_count}   |   Depth FPS: {self.depth_count}"
        )
        self.rgb_count = 0
        self.depth_count = 0

    def process_frame(self):
        # 아직 Nav2 이동 중이거나, 사분면 네비 단계면 추종 로직 돌리지 않음
        if self.phase != 'SEARCH_FOLLOW':
            return

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
            boxes = self.run_follow_yolo(rgb_display)

            target_found = False
            target_cx = None
            target_cy = None
            target_dist = None

            MIN_CONF = 0.9
            best_box = None
            best_conf = 0.0

            # target_class 중에서 가장 confidence 높은 박스 추출
            for (x1, y1, x2, y2, name, conf) in boxes:
                if name == self.target_class and conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2, name, conf)

            # best_box만 시각화
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

            # rqt용 이미지 발행
            img_msg = self.bridge.cv2_to_imgmsg(rgb_display, encoding='bgr8')
            img_msg.header.stamp = now.to_msg()
            img_msg.header.frame_id = 'oakd_rgb_frame'
            self.yolo_image_pub.publish(img_msg)

            # Depth 기반 거리 계산 (바운딩 박스 하단에서 조금 위쪽)
            if best_box is not None and best_conf >= MIN_CONF and depth is not None:
                x1, y1, x2, y2, name, conf = best_box
                cx = int((x1 + x2) / 2)
                cy = int(y2 - (y2 - y1) * 0.05)  # 아래에서 5% 위쪽

                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    z = float(depth[cy, cx]) / 1000.0  # mm → m
                    if 0.2 < z < 5.0:
                        target_found = True
                        target_cx = cx
                        target_cy = cy
                        target_dist = z
            else:
                if best_box is not None and best_conf < MIN_CONF:
                    self.get_logger().debug(
                        f"Detected '{self.target_class}' but conf={best_conf:.2f} < {MIN_CONF}"
                    )

            # 상태머신: TRACKING / SEARCHING
            if target_found:
                self.last_detection_time = now
                self.state = "TRACKING"
                self.search_start_time = None
                self.track_target(target_cx, target_cy, target_dist, rgb.shape)
            else:
                if self.state != "SEARCHING":
                    self.state = "SEARCHING"
                    self.search_start_time = now
                    self.get_logger().info(f"{self.target_class}를 찾지 못했습니다. 360도 회전합니다.")
                self.search_for_target(now)
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
        """Depth 기반으로 target을 일정 거리 두고 따라가기"""
        height, width, _ = image_shape

        center_x = width / 2.0
        error_x = (cx - center_x) / center_x  # -1 ~ +1

        dist_error = dist - self.follow_distance

        if abs(dist_error) < self.dist_deadband:
            dist_error = 0.0

        if abs(error_x) < self.angle_deadband:
            error_x = 0.0

        linear_x = self.k_v * dist_error
        if dist < self.follow_distance and dist_error <= 0:
            linear_x = 0.0

        angular_z = - self.k_w * error_x

        linear_x = max(min(linear_x, self.max_linear_speed), -self.max_linear_speed)
        angular_z = max(min(angular_z, self.max_angular_speed), -self.max_angular_speed)

        if linear_x == 0.0 and angular_z == 0.0:
            self.stop_robot()
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def search_for_target(self, now):
        """Target이 안 보일 때 제자리 360도 회전하며 찾기"""
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
                f"{self.target_class}를 찾지 못했습니다. IDLE 상태로 돌아갑니다."
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

    def run_follow_yolo(self, rgb_image):
        results = self.follow_yolo(rgb_image)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.follow_yolo.names[cls_id]
                boxes.append((x1, y1, x2, y2, cls_name, conf))
        return boxes

    # ---------------------------
    # 종료 처리
    # ---------------------------
    def shutdown_node(self):
        self.get_logger().info("노드 종료 중...")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = NavAndFollow()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt로 종료")
    finally:
        node.shutdown_node()


if __name__ == "__main__":
    main()