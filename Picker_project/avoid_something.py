import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav2_simple_commander.robot_navigator import TaskResult
import time
import threading
import cv2 # OpenCV
from cv_bridge import CvBridge, CvBridgeError # ROS->OpenCV 변환기
from ultralytics import YOLO # YOLO 모델

# =========================================
# 1. 안전 가드 + YOLO 탐지기
# =========================================
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        
        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # LiDAR 및 제어        
        self.scan_sub = self.create_subscription(LaserScan, '/robot3/scan', self.scan_callback, qos)
        self.input_sub = self.create_subscription(Twist, '/cmd_vel_input', self.input_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot3/cmd_vel', 10)

        self.img_sub = self.create_subscription(Image, '/robot3/oakd/rgb/preview/image_raw', self.img_callback, qos)
        
        self.bridge = CvBridge()
        self.latest_cv_image = None # 가장 최근 이미지 저장용
        
        print("📦 YOLO 모델을 불러오는 중...", flush=True)
        # 팀원분이 준 모델 경로 (경로가 틀리면 에러나니 확인 필수!)
        try:
            self.model = YOLO("/home/rokey/rokey_ws/src/final_project/box_yolo8n.pt")
            print("✅ YOLO 모델 로드 완료!", flush=True)
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}", flush=True)
            self.model = None

        # 상태 변수들
        self.emergency_dist = 0.40 
        self.current_dist = 10.0
        self.is_danger = False
        self.phase2_active = False 
        self.obstacle_dir = 1.0
        self.is_sensor_active = False

    def scan_callback(self, msg):
        self.is_sensor_active = True
        ranges = msg.ranges
        count = len(ranges)
        if count == 0: return

        # 전방 각도
        fov_ratio = 45 / 360
        split_idx = int(count * fov_ratio) 
        half_idx = split_idx // 2
        
        left_slice = ranges[0 : half_idx]
        right_slice = ranges[-half_idx : ]
        
        # 노이즈 필터링 (0.12m ~ 1.0m)
        valid_left = [r for r in left_slice if 0.12 < r < 1.0]
        valid_right = [r for r in right_slice if 0.12 < r < 1.0]
        
        min_left = min(valid_left) if valid_left else 10.0
        min_right = min(valid_right) if valid_right else 10.0
        min_dist = min(min_left, min_right)

        self.current_dist = min_dist
        self.is_danger = (min_dist < self.emergency_dist)

        if min_right < min_left: self.obstacle_dir = 1.0 
        else: self.obstacle_dir = -1.0

    def img_callback(self, msg):
        # 카메라 데이터를 받을 때마다 OpenCV 형식으로 변환해서 저장해둠
        try:
            # ROS Image -> OpenCV Image (BGR)
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            pass

    def input_callback(self, msg):
        if not self.phase2_active: return
        final_cmd = Twist()
        if self.is_danger:
            final_cmd.linear.x = 0.0
            final_cmd.angular.z = 0.5 * self.obstacle_dir
        else:
            final_cmd = msg
        self.cmd_vel_pub.publish(final_cmd)

    # [추가] 물체 개수 세기 함수
    def detect_and_count(self):
        if self.model is None:
            print("⚠️ 모델이 없어서 탐지 불가.")
            return -1
        
        if self.latest_cv_image is None:
            print("⚠️ 카메라 이미지가 아직 안 들어옴.")
            return -1

        print("📸 이미지 분석 중...", flush=True)
        # YOLO 추론 (verbose=False는 로그 끄기)
        results = self.model(self.latest_cv_image, verbose=False)[0]
        
        # 박스 개수 세기
        box_count = len(results.boxes)
        
        # (선택 사항) 결과 이미지를 화면에 띄우고 싶다면 아래 주석 해제
        # res_plotted = results.plot()
        # cv2.imshow("YOLO Result", res_plotted)
        # cv2.waitKey(2000) # 2초간 보여줌
        # cv2.destroyAllWindows()
        
        return box_count

# =========================================
# 2. 메인 실행 로직
# =========================================
def main():
    rclpy.init()
    
    safety_node = SafetyMonitor()
    navigator = TurtleBot4Navigator()

    # 백그라운드 실행 (센서 & 카메라 수신)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(safety_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # --- 초기화 ---
    if not navigator.getDockedStatus(): navigator.dock()
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    navigator.undock()

    print("⏳ 센서 연결 확인 중...", flush=True)
    while not safety_node.is_sensor_active:
        time.sleep(0.1)
    print("✅ 센서 정상 연결됨.", flush=True)

    config_cli = safety_node.create_client(SetParameters, '/robot3/controller_server/set_parameters')
    def set_nav2_params(max_speed, xy_tol, yaw_tol):
        if not config_cli.wait_for_service(timeout_sec=1.0): return
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='FollowPath.max_vel_x', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=max_speed)),
            Parameter(name='FollowPath.xy_goal_tolerance', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=xy_tol)),
            Parameter(name='FollowPath.yaw_goal_tolerance', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=yaw_tol))
        ]
        config_cli.call_async(req)
        time.sleep(0.5)

    # ---------------------------------------------------------
    # [핵심 수정] goToPose를 사용하여 쓰레드 없이 비동기 이동
    # ---------------------------------------------------------
    def drive_smart(target_pose, arrival_radius, strict_mode=False):
        mode_str = "정밀" if strict_mode else "고속"
        print(f"🚗 [{mode_str}] 이동 시작!", flush=True)
        
        # [변경점] startToPose(Blocking) 대신 goToPose(Non-blocking) 사용!
        # 쓰레드를 만들 필요가 없어졌습니다.
        navigator.goToPose(target_pose)

        last_known_dist = float('inf')

        while not navigator.isTaskComplete():
            
            # (A) 위험 감지
            if safety_node.is_danger:
                print(f"🚨 [장애물] {safety_node.current_dist:.2f}m 감지 -> 회피!", flush=True)
                navigator.cancelTask() # Nav2 중단
                
                # 정지 및 후진
                stop_twist = Twist(); stop_twist.linear.x = -0.15
                safety_node.cmd_vel_pub.publish(stop_twist); time.sleep(0.5)
                
                print("🔄 회피 회전 중...", flush=True)
                while safety_node.is_danger:
                    twist = Twist(); twist.linear.x = 0.0
                    twist.angular.z = 0.6 * safety_node.obstacle_dir 
                    safety_node.cmd_vel_pub.publish(twist)
                    time.sleep(0.1)
                
                print("✅ 안전 확보. 재출발.", flush=True)
                safety_node.cmd_vel_pub.publish(Twist()); time.sleep(0.5)
                return "RETRY"

            # (B) 도착 체크
            feedback = navigator.getFeedback()
            if feedback:
                dist = feedback.distance_remaining
                last_known_dist = dist
                if not strict_mode and dist < arrival_radius:
                    print(f"🚩 [도착] 반경 진입 ({dist:.2f}m).", flush=True)
                    navigator.cancelTask(); safety_node.cmd_vel_pub.publish(Twist())
                    return "SUCCESS"
            
            time.sleep(0.05)

        # 결과 확인
        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED: return "SUCCESS"
        elif result == TaskResult.CANCELED: return "RETRY"
        
        limit = arrival_radius + 0.05 if strict_mode else arrival_radius + 0.3
        return "SUCCESS" if last_known_dist < limit else "FAIL"

    def nudge_robot(distance_m, speed_mps=0.05):
        print(f"📏 [마무리] {distance_m}m 전진...", flush=True)
        duration = distance_m / speed_mps
        twist = Twist(); twist.linear.x = speed_mps
        start_time = time.time()
        while (time.time() - start_time) < duration:
            safety_node.cmd_vel_pub.publish(twist); time.sleep(0.1)
        safety_node.cmd_vel_pub.publish(Twist())

    # =========================================================
    # Phase 1
    # =========================================================
    goal_1 = navigator.getPoseStamped([-5.9, 0.4], TurtleBot4Directions.SOUTH)
    set_nav2_params(0.31, 0.5, 3.14)
    
    while True:
        status = drive_smart(goal_1, arrival_radius=1.0, strict_mode=False)
        if status == "SUCCESS": print("✅ 1차 완료.", flush=True); break
        elif status == "RETRY": continue
        else: print("❌ 1차 실패.", flush=True); rclpy.shutdown(); return

    # =========================================================
    # Phase 2
    # =========================================================
    goal_2 = navigator.getPoseStamped([-6.38, 1.8], TurtleBot4Directions.SOUTH)
    set_nav2_params(0.1, 0.05, 0.1)
    
    print("🐢 정밀 모드...", flush=True)
    while True:
        status = drive_smart(goal_2, arrival_radius=0.05, strict_mode=True)
        if status == "SUCCESS": 
            print("🎉 최종 완료!", flush=True)
            nudge_robot(0.05)
            break
        elif status == "RETRY": continue
        else: print("❌ 최종 실패.", flush=True); rclpy.shutdown(); return

    # =========================================================
    # [NEW] Phase 3: YOLO 탐지 및 개수 세기
    # =========================================================
    print("\n=== [Phase 3] 물체 감지 시작 ===", flush=True)
    
    # 이미지가 들어올 때까지 잠깐 대기 (카메라 안정화)
    time.sleep(2.0)
    
    # 여기서 탐지 함수 호출!
    box_count = safety_node.detect_and_count()
    
    print(f"\n📦📦📦 [결과] 감지된 박스 개수: {box_count} 개 📦📦📦\n", flush=True)


    # =========================================================
    # Phase 4: 추적 모드 (기존 Phase 2)
    # =========================================================
    print("\n=== [Phase 4] 추적 모드 전환 ===", flush=True)
    print("👉 '/cmd_vel_input' 대기 중...", flush=True)
    
    safety_node.phase2_active = True
    try:
        while rclpy.ok(): time.sleep(1)
    except KeyboardInterrupt: pass
    finally:
        safety_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()