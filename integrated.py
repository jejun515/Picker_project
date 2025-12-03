import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions
from ultralytics import YOLO

from tf_transformations import quaternion_from_euler
import numpy as np
import cv2
import threading
import math


# 웹캠에서 탐지할 객체 클래스 ID
DETECT_TARGET = [0]  # 0: blue_car, 1: green_car

# 각 사분면에 해당하는 목표 좌표 (x, y, theta)
QUADRANT_TARGET_POSES = {
    1: (0.00464523, 3.32853, math.pi / 2),
    2: (-2.37742, 4.07556, 3 * math.pi / 4),
    3: (-4.30382, 1.58606, -3 * math.pi / 4),
    4: (-1.51404, -0.197009, -math.pi / 2)
}


class IntegratedRobotTracker(Node):
    def __init__(self):
        super().__init__('integrated_robot_tracker')
        self.get_logger().info('[INFO] IntegratedRobotTracker 노드가 실행되었습니다.')

        # ===== 공통 설정 =====
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ===== 웹캠 설정 (MoveRobot 파트) =====
        self.webcam_model = YOLO('/home/rokey/Picker_project/mixed_results.pt')
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("[ERROR] 웹캠을 열 수 없습니다.")
            raise IOError("Webcam failed to open.")

        # 웹캠 annotated 이미지 퍼블리셔
        self.webcam_image_publisher = self.create_publisher(
            Image, '/webcam/annotated_frame', 10
        )

        # Nav2 액션 클라이언트
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        self.previous_quadrant = -1
        self.navigation_goal_handle = None
        self.navigation_in_progress = False

        # ===== OAK-D 카메라 설정 (DepthToMap 파트) =====
        self.qos_sensor = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'

        self.get_logger().info("Loading OAK-D YOLO model...")
        self.oakd_yolo = YOLO('/home/rokey/Picker_project/yolo_mixed.pt')
        self.get_logger().info("OAK-D YOLO loaded.")

        self.target_class = "customer_b"

        self.K = None
        self.depth_image = None
        self.rgb_image = None
        self.yolo_running = False

        # OAK-D YOLO 이미지 퍼블리셔
        self.qos_image = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.oakd_yolo_image_pub = self.create_publisher(
            Image, 'image_yolo', self.qos_image
        )

        # cmd_vel 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{ns}/cmd_vel', 10
        )

        # ===== 추적 파라미터 =====
        self.follow_distance = 1.1
        self.k_v = 0.8
        self.k_w = 1.2
        self.max_linear_speed = 0.25
        self.max_angular_speed = 0.5
        self.dist_deadband = 0.05
        self.angle_deadband = 0.17

        self.lost_timeout = 1.0
        self.search_angular_speed = 0.5
        self.search_duration = 2 * math.pi / abs(self.search_angular_speed)

        # ===== 상태 관리 =====
        # 상태: "WEBCAM_DETECTION", "NAVIGATING", "OAKD_SEARCHING", "OAKD_TRACKING"
        self.state = "WEBCAM_DETECTION"
        self.last_detection_time = None
        self.search_start_time = None

        # 🔹 OAK-D 추적 완전 상실 타이머 (타겟이 시야를 완전히 벗어난 경우 판단)
        self.oakd_lost_timeout = 3.0  # 3초간 미검출 시 웹캠 모드로 복귀

        # ===== TurtleBot4 네비게이터 초기화 =====
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

        # ===== OAK-D 카메라 서브스크립션 =====
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

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        # ===== 타이머 설정 =====
        # 웹캠 검출 루프 (0.5초마다)
        self.webcam_timer = self.create_timer(0.5, self.webcam_detection_loop)
        
        # OAK-D 처리는 5초 후 시작 (TF Tree 안정화)
        self.get_logger().info("TF Tree 안정화 대기 중... 5초 후 OAK-D 활성화")
        self.start_timer = self.create_timer(5.0, self.start_oakd_processing)

    # ========================================
    # OAK-D 초기화 함수
    # ========================================
    def start_oakd_processing(self):
        self.get_logger().info("TF Tree 안정화 완료. OAK-D 처리 시작.")
        self.oakd_timer = self.create_timer(0.2, self.oakd_process_frame)
        self.start_timer.cancel()

    # ========================================
    # 웹캠 관련 함수들 (MoveRobot 파트)
    # ========================================
    def get_quadrant(self, xc, yc, frame_cx, frame_cy):
        """이미지 중심 기준으로 사분면 계산"""
        if (xc >= frame_cx) and (yc <= frame_cy):
            return 1
        elif (xc <= frame_cx) and (yc <= frame_cy):
            return 2
        elif (xc <= frame_cx) and (yc >= frame_cy):
            return 3
        else:
            return 4

    def send_nav2_goal(self, x, y, theta):
        """Nav2 목표 좌표로 NavigateToPose 전송"""
        if not self._action_client.wait_for_server(timeout_sec=1.0):
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

        self.get_logger().info(f'[INFO] 새로운 목표 좌표로 이동: ({x:.2f}, {y:.2f})')
        
        # 비동기로 goal 전송 및 결과 콜백 등록
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )
        send_goal_future.add_done_callback(self.navigation_goal_response_callback)
        
        self.navigation_in_progress = True
        self.state = "NAVIGATING"

    def navigation_goal_response_callback(self, future):
        """Navigation goal이 수락되었는지 확인"""
        self.navigation_goal_handle = future.result()
        if not self.navigation_goal_handle.accepted:
            self.get_logger().warn('[WARN] Navigation goal이 거부되었습니다.')
            self.navigation_in_progress = False
            self.state = "WEBCAM_DETECTION"
            return

        self.get_logger().info('[INFO] Navigation goal이 수락되었습니다.')
        
        # 결과 대기
        result_future = self.navigation_goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """Navigation 완료 시 호출"""
        result = future.result().result
        status = future.result().status
        
        self.navigation_in_progress = False
        
        if status == 4:  # SUCCEEDED
            self.get_logger().info('[INFO] 목표 지점 도착 완료! OAK-D 탐색 모드로 전환합니다.')
            self.state = "OAKD_SEARCHING"
            self.search_start_time = None
        else:
            self.get_logger().warn(f'[WARN] Navigation 실패 (status: {status}). 웹캠 모드로 복귀.')
            self.state = "WEBCAM_DETECTION"

    def webcam_detection_loop(self):
        """웹캠에서 YOLO 검출 및 사분면 판단"""
        # 🔹 OAK-D 추적 중(OAKD_TRACKING)이면 웹캠 검출은 하되 Navigation 명령은 보내지 않음
        # 🔹 NAVIGATING 또는 OAKD_SEARCHING 중에는 사분면 변경 시 즉시 재설정
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("[WARN] 웹캠 프레임을 읽을 수 없습니다.")
            return

        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        results = self.webcam_model.predict(
            frame, classes=DETECT_TARGET, conf=0.5, verbose=False
        )
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

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (xc, yc), 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame, f'Q{new_quadrant}',
                    (xc, yc + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                break

        # Annotated 이미지 퍼블리시
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.webcam_image_publisher.publish(ros_image_msg)
        except Exception as e:
            self.get_logger().error(f'[ERROR] Webcam frame 발행 실패: {e}')

        # 🔹 사분면 변경 감지 및 Navigation 중단/재시작
        # OAKD_TRACKING 상태일 때는 웹캠 명령을 무시 (OAK-D 추적 우선)
        if new_quadrant != -1 and new_quadrant != self.previous_quadrant:
            self.get_logger().info(
                f"[INFO] 웹캠 사분면 변경 감지: {self.previous_quadrant} -> {new_quadrant}"
            )
            
            # OAKD_TRACKING 중이면 웹캠 명령 무시
            if self.state == "OAKD_TRACKING":
                self.get_logger().info(
                    "[INFO] OAK-D 추적 중이므로 웹캠 사분면 변경 무시 (추적 우선)"
                )
                # previous_quadrant는 업데이트하지 않음 (추적 끝난 후 비교를 위해)
                return
            
            # NAVIGATING 또는 OAKD_SEARCHING 중이면 즉시 중단하고 재설정
            if self.state in ["NAVIGATING", "OAKD_SEARCHING"]:
                self.get_logger().info("[INFO] 기존 동작 중단. 새로운 목표로 재설정.")
                # Navigation 취소
                if self.navigation_goal_handle is not None:
                    self.navigation_goal_handle.cancel_goal_async()
                
                # 로봇 정지
                self.stop_robot()
            
            # 새로운 목표로 Navigation 시작
            x, y, theta = QUADRANT_TARGET_POSES[new_quadrant]
            self.send_nav2_goal(x, y, theta)
            self.previous_quadrant = new_quadrant

    # ========================================
    # OAK-D 카메라 콜백 함수들
    # ========================================
    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f"Camera intrinsics: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}"
                )
                self.logged_intrinsics = True

    def depth_callback(self, msg):
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = now_sec - msg_sec
        
        if dt > 0.5:
            self.get_logger().warn(f"Depth frame too old ({dt:.2f}s). Dropping.")
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if depth is None or depth.size == 0:
                self.get_logger().error("Depth image is empty")
            else:
                if not self.logged_depth_shape:
                    self.get_logger().info(f"Depth image: {depth.shape}")
                    self.logged_depth_shape = True

            with self.lock:
                self.depth_image = depth

        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def rgb_callback(self, msg):
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = now_sec - msg_sec

        if dt > 0.5:
            self.get_logger().warn(f"RGB frame too old ({dt:.2f}s). Dropping.")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if rgb is None or rgb.size == 0:
                self.get_logger().error("Decoded RGB image is empty")
            else:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f"RGB image: {rgb.shape}")
                    self.logged_rgb_shape = True

            with self.lock:
                self.rgb_image = rgb

        except Exception as e:
            self.get_logger().error(f"RGB decode failed: {e}")

    # ========================================
    # OAK-D 프레임 처리 (YOLO + 추적)
    # ========================================
    def oakd_process_frame(self):
        """OAK-D 카메라로 타겟 탐색 및 추적"""
        # OAKD_SEARCHING 또는 OAKD_TRACKING 상태일 때만 처리
        if self.state not in ["OAKD_SEARCHING", "OAKD_TRACKING"]:
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
            boxes = self.run_oakd_yolo(rgb_display)

            target_found = False
            target_cx = None
            target_cy = None
            target_dist = None

            MIN_CONF = 0.9

            best_box = None
            best_conf = 0.0

            # 타겟 클래스에서 가장 높은 confidence 박스 찾기
            for (x1, y1, x2, y2, name, conf) in boxes:
                if name == self.target_class and conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2, name, conf)

            # 시각화
            if best_box is not None and best_conf >= MIN_CONF:
                x1, y1, x2, y2, name, conf = best_box
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    rgb_display, f"{name} {conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # YOLO 결과 이미지 퍼블리시
            img_msg = self.bridge.cv2_to_imgmsg(rgb_display, encoding='bgr8')
            img_msg.header.stamp = now.to_msg()
            img_msg.header.frame_id = 'oakd_rgb_frame'
            self.oakd_yolo_image_pub.publish(img_msg)

            # 거리 계산
            if best_box is not None and best_conf >= MIN_CONF and depth is not None:
                x1, y1, x2, y2, name, conf = best_box
                cx = int((x1 + x2) / 2)
                cy = int(y2 - (y2 - y1) * 0.05)

                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    z = float(depth[cy, cx]) / 1000.0
                    if 0.2 < z < 5.0:
                        target_found = True
                        target_cx = cx
                        target_cy = cy
                        target_dist = z

            # 🔹 타겟 발견 시 추적 모드로 전환 (웹캠 명령보다 우선)
            if target_found:
                self.last_detection_time = now
                
                # 탐색 중에서 추적으로 전환
                if self.state == "OAKD_SEARCHING":
                    self.get_logger().info("[INFO] OAK-D에서 타겟 발견! 추적 모드로 전환")
                
                self.state = "OAKD_TRACKING"
                self.search_start_time = None
                self.track_target(target_cx, target_cy, target_dist, rgb.shape)
                
            else:
                # 🔹 타겟 미발견 처리
                if self.state == "OAKD_SEARCHING":
                    # 계속 360도 회전 탐색
                    self.search_for_target(now)
                    
                elif self.state == "OAKD_TRACKING":
                    # 🔹 추적 중 타겟을 놓친 경우
                    if self.last_detection_time is None:
                        # 이전에도 없었으면 탐색 모드로
                        self.get_logger().info("[INFO] 타겟 미발견. 탐색 모드로 전환")
                        self.state = "OAKD_SEARCHING"
                        self.search_start_time = None
                        self.stop_robot()
                    else:
                        elapsed = (now - self.last_detection_time).nanoseconds * 1e-9
                        
                        if elapsed < self.lost_timeout:
                            # 1초 이내: 잠시 대기 (추적 유지)
                            self.stop_robot()
                        elif elapsed < self.oakd_lost_timeout:
                            # 1~3초: 제자리 회전으로 재탐색 시도
                            if self.search_start_time is None:
                                self.get_logger().info(
                                    "[INFO] 타겟 놓침. 제자리 회전으로 재탐색 시작"
                                )
                                self.search_start_time = now
                            self.search_for_target(now)
                        else:
                            # 🔹 3초 이상: 완전히 시야를 벗어남 → 웹캠 모드로 복귀
                            self.get_logger().info(
                                "[INFO] 타겟이 OAK-D 시야를 완전히 벗어남. "
                                "웹캠 모드로 복귀하여 재탐지 시작"
                            )
                            self.state = "WEBCAM_DETECTION"
                            self.stop_robot()
                            self.last_detection_time = None
                            self.search_start_time = None

        except Exception as e:
            self.get_logger().warn(f"OAK-D frame processing error: {e}")
        finally:
            self.yolo_running = False

    def track_target(self, cx, cy, dist, image_shape):
        """타겟을 일정 거리 유지하며 추적"""
        height, width, _ = image_shape

        center_x = width / 2.0
        error_x = (cx - center_x) / center_x

        dist_error = dist - self.follow_distance

        if abs(dist_error) < self.dist_deadband:
            dist_error = 0.0

        if abs(error_x) < self.angle_deadband:
            error_x = 0.0

        linear_x = self.k_v * dist_error
        if dist < self.follow_distance and dist_error <= 0:
            linear_x = 0.0

        angular_z = -self.k_w * error_x

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
        """제자리에서 360도 회전하며 타겟 탐색"""
        if self.search_start_time is None:
            self.search_start_time = now
            self.get_logger().info("[INFO] OAK-D 360도 회전 탐색 시작")

        elapsed = (now - self.search_start_time).nanoseconds * 1e-9

        if elapsed < self.search_duration:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = float(self.search_angular_speed)
            self.cmd_vel_pub.publish(twist)
        else:
            self.get_logger().info(
                "[INFO] 360도 탐색 완료. 타겟 미발견. 웹캠 모드로 복귀."
            )
            self.state = "WEBCAM_DETECTION"
            self.stop_robot()
            self.last_detection_time = None
            self.search_start_time = None

    def stop_robot(self):
        """로봇 정지"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def run_oakd_yolo(self, rgb_image):
        """OAK-D 이미지에서 YOLO 검출"""
        results = self.oakd_yolo(rgb_image)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.oakd_yolo.names[cls_id]
                boxes.append((x1, y1, x2, y2, cls_name, conf))
        return boxes

    def destroy_node(self):
        """노드 종료 시 정리"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = IntegratedRobotTracker()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except IOError:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Main exception: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()