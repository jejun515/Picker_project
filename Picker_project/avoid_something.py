import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import LaserScan, Image
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav2_simple_commander.robot_navigator import TaskResult
import time
import threading
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

# =========================================
# 1. 안전 가드 + 통신 모듈
# =========================================
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 센서 및 제어
        ns = self.get_namespace()
        self.scan_sub = self.create_subscription(LaserScan, f'{ns}/scan', self.scan_callback, qos)
        self.input_sub = self.create_subscription(Twist, f'/cmd_vel_input', self.input_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'{ns}/cmd_vel', 10)
        self.img_sub = self.create_subscription(Image, f'{ns}/oakd/rgb/preview/image_raw', self.img_callback, qos)
        
        # [NEW] 팀원 코드(OrderManager)와 연결되는 토픽
        self.order_sub = self.create_subscription(PoseArray, '{ns}/box_order_goals', self.order_callback, 10)
        
        self.bridge = CvBridge()
        self.latest_cv_image = None
        
        # YOLO 로드
        print("📦 YOLO 모델 로딩 중...", flush=True)
        try:
            self.model = YOLO("/home/rokey/rokey_ws/src/final_project/box_yolo8n.pt")
            print("✅ YOLO 로드 완료.", flush=True)
        except Exception:
            self.model = None

        self.emergency_dist = 0.40 
        self.current_dist = 10.0
        self.is_danger = False
        self.phase2_active = False 
        self.obstacle_dir = 1.0
        self.is_sensor_active = False
        
        # 좌표 수신 상태
        self.received_poses = []
        self.has_new_order = False

    def scan_callback(self, msg):
        self.is_sensor_active = True
        ranges = msg.ranges
        count = len(ranges)
        if count == 0: return

        CENTER_RATIO = 0.25 
        center_idx = int(count * CENTER_RATIO)
        fov_ratio = 30 / 360
        half_width = int(count * fov_ratio / 2)
        
        start_idx = max(0, center_idx - half_width)
        end_idx = min(count, center_idx + half_width)
        
        front_ranges = ranges[start_idx : end_idx]
        valid_ranges = [r for r in front_ranges if 0.18 < r < 1.0]
        min_dist = min(valid_ranges) if valid_ranges else 10.0

        self.current_dist = min_dist
        self.is_danger = (min_dist < self.emergency_dist)
        
        mid = len(front_ranges) // 2
        l_val = min([r for r in front_ranges[:mid] if r > 0.18], default=10.0)
        r_val = min([r for r in front_ranges[mid:] if r > 0.18], default=10.0)
        if r_val < l_val: self.obstacle_dir = 1.0 
        else: self.obstacle_dir = -1.0

    def img_callback(self, msg):
        try: self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: pass

    def input_callback(self, msg):
        if not self.phase2_active: return
        final_cmd = Twist()
        if self.is_danger:
            final_cmd.linear.x = 0.0
            final_cmd.angular.z = 0.5 * self.obstacle_dir
        else: final_cmd = msg
        self.cmd_vel_pub.publish(final_cmd)

    # [NEW] 주문 수신 콜백
    def order_callback(self, msg):
        # 메시지가 [박스위치, 도착지위치] 2개가 들어와야 함
        if len(msg.poses) >= 2:
            self.received_poses = msg.poses
            self.has_new_order = True
            p1 = msg.poses[0].position
            p2 = msg.poses[1].position
            print(f"\n📨 [주문 수신] 박스: ({p1.x:.2f}, {p1.y:.2f}) -> 도착지: ({p2.x:.2f}, {p2.y:.2f})", flush=True)

    def detect_and_count(self):
        if self.model is None or self.latest_cv_image is None: return -1
        print("📸 YOLO 분석 중...", flush=True)
        results = self.model(self.latest_cv_image, verbose=False)[0]
        return len(results.boxes)

# =========================================
# 2. 메인 실행 로직
# =========================================
def main():
    rclpy.init()
    
    safety_node = SafetyMonitor()
    navigator = TurtleBot4Navigator()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(safety_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # -------------------------------------------------------------
    # [대기 모드] 팀원 코드(OrderManager)에서 주문이 올 때까지 대기
    # -------------------------------------------------------------
    print("\n🌐 [대기 중] '/robot<ns>/box_order_goals' 토픽을 기다리는 중...", flush=True)
    while not safety_node.has_new_order:
        time.sleep(1.0)    
    
    if not navigator.getDockedStatus(): navigator.dock()
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    navigator.undock()

    print("⏳ 센서 확인 중...", flush=True)
    while not safety_node.is_sensor_active: time.sleep(0.1)
    print("✅ 센서 연결됨.", flush=True)


    
    # 좌표 추출
    box_pose_raw = safety_node.received_poses[0]
    room_pose_raw = safety_node.received_poses[1]
    
    # Phase 2 목표 (박스 위치)
    target_box_x = box_pose_raw.position.x
    target_box_y = box_pose_raw.position.y
    
    # Phase 4 목표 (도착지)
    target_room_x = room_pose_raw.position.x
    target_room_y = room_pose_raw.position.y
    
    print(f"🚀 미션 시작! 1차목표: ({target_box_x}, {target_box_y})", flush=True)

    # Nav2 파라미터 설정 클라이언트
    ns = safety_node.get_namespace()
    config_cli = safety_node.create_client(SetParameters, f'{ns}/controller_server/set_parameters')
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

    def drive_smart(target_pose, arrival_radius, strict_mode=False):
        mode_str = "정밀" if strict_mode else "고속"
        print(f"🚗 [{mode_str}] 이동 -> {target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f}", flush=True)
        
        navigator.goToPose(target_pose)
        print("⏳ 경로 계산...", flush=True)
        time.sleep(2.0) 

        last_known_dist = float('inf')

        while not navigator.isTaskComplete():
            if safety_node.is_danger:
                print(f"🚨 [장애물] {safety_node.current_dist:.2f}m -> 회피!", flush=True)
                navigator.cancelTask()
                stop_twist = Twist(); stop_twist.linear.x = -0.15
                safety_node.cmd_vel_pub.publish(stop_twist); time.sleep(0.5)
                
                print("🔄 회피 중...", flush=True)
                while safety_node.is_danger:
                    twist = Twist(); twist.linear.x = 0.0
                    twist.angular.z = 0.6 * safety_node.obstacle_dir 
                    safety_node.cmd_vel_pub.publish(twist)
                    time.sleep(0.1)
                
                print("✅ 재출발.", flush=True)
                safety_node.cmd_vel_pub.publish(Twist()); time.sleep(0.5)
                return "RETRY"

            feedback = navigator.getFeedback()
            if feedback:
                dist = feedback.distance_remaining
                last_known_dist = dist
                if not strict_mode and dist < arrival_radius:
                    print(f"🚩 [도착] 반경 진입 ({dist:.2f}m).", flush=True)
                    navigator.cancelTask(); safety_node.cmd_vel_pub.publish(Twist())
                    return "SUCCESS"
            time.sleep(0.05)

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED: return "SUCCESS"
        elif result == TaskResult.CANCELED: return "RETRY"
        limit = arrival_radius + 0.05 if strict_mode else arrival_radius + 0.3
        return "SUCCESS" if last_known_dist < limit else "FAIL"

    def nudge_robot(distance_m, speed_mps=0.05):
        action = "전진" if distance_m > 0 else "후진"
        print(f"📏 [마무리] {abs(distance_m)}m {action}...", flush=True)
        duration = abs(distance_m) / speed_mps
        twist = Twist(); twist.linear.x = speed_mps if distance_m > 0 else -speed_mps
        start_time = time.time()
        while (time.time() - start_time) < duration:
            safety_node.cmd_vel_pub.publish(twist); time.sleep(0.1)
        safety_node.cmd_vel_pub.publish(Twist())

    # =========================================================
    # Phase 1: 1차 진입 (고정 좌표 사용)
    # =========================================================
    # 1차 진입 지점은 보통 고정되어 있으므로 그대로 둠
    goal_1 = navigator.getPoseStamped([-5.9, 0.4], TurtleBot4Directions.SOUTH)
    set_nav2_params(0.31, 0.5, 3.14)
    
    while True:
        status = drive_smart(goal_1, arrival_radius=1.0, strict_mode=False)
        if status == "SUCCESS": print("✅ 1차 진입 완료.", flush=True); break
        elif status == "RETRY": continue
        else: print("❌ 1차 실패.", flush=True); rclpy.shutdown(); return

    # =========================================================
    # Phase 2: 박스 위치로 이동 (수신된 좌표 사용)
    # =========================================================
    print("📉 [접근] 안전거리 15cm로 축소.", flush=True)
    safety_node.emergency_dist = 0.15 
    
    # [수신된 박스 좌표 사용]
    goal_2 = navigator.getPoseStamped([target_box_x, target_box_y], TurtleBot4Directions.SOUTH)
    set_nav2_params(0.1, 0.05, 0.1)
    
    while True:
        status = drive_smart(goal_2, arrival_radius=0.05, strict_mode=True)
        if status == "SUCCESS": 
            print("🎉 박스 앞 도착!", flush=True)
            nudge_robot(0.15) 
            break
        elif status == "RETRY": continue
        else: print("❌ 도착 실패.", flush=True); rclpy.shutdown(); return

    # =========================================================
    # Phase 3: YOLO 탐지
    # =========================================================
    print("\n=== [Phase 3] 물체 감지 시작 ===", flush=True)
    time.sleep(2.0)
    box_count = safety_node.detect_and_count()
    print(f"\n📦📦📦 [결과] 감지된 박스 개수: {box_count} 개 📦📦📦\n", flush=True)

    print("🔙 후진하여 거리 확보.", flush=True)
    nudge_robot(-0.25)
    print("📈 [복구] 안전거리 0.5m로 복구.", flush=True)
    safety_node.emergency_dist = 0.40

    # =========================================================
    # Phase 4: 도착지로 이동 (수신된 좌표 사용)
    # =========================================================
    print("\n=== [Phase 4] 도착지로 이동 ===", flush=True)
    # [수신된 도착지 좌표 사용]
    goal_3 = navigator.getPoseStamped([target_room_x, target_room_y], TurtleBot4Directions.WEST)
    set_nav2_params(0.31, 0.5, 0.5) 

    while True:
        status = drive_smart(goal_3, arrival_radius=0.2, strict_mode=False)
        if status == "SUCCESS": 
            print("✅ 2차 지점 도착 완료!", flush=True)
            break
        elif status == "RETRY": continue
        else: print("❌ 이동 실패.", flush=True); rclpy.shutdown(); return
    time.sleep(5.0)

    # =========================================================
    # Phase 5: 도킹 복귀
    # =========================================================
    print("\n=== [Phase 5] 도킹 스테이션 복귀 ===", flush=True)
    dock_pose = navigator.getPoseStamped([-0.26, -0.3], TurtleBot4Directions.NORTH)
    set_nav2_params(0.31, 0.1, 0.1)

    while True:
        status = drive_smart(dock_pose, arrival_radius=0.10, strict_mode=True)
        if status == "SUCCESS": print("✅ 도킹 준비 위치 도착.", flush=True); break
        elif status == "RETRY": continue
        else: print("❌ 복귀 실패.", flush=True); rclpy.shutdown(); return

    print("🔋 도킹 시퀀스 시작...", flush=True)
    navigator.dock()

    if navigator.getDockedStatus(): print("🎉 도킹 성공! 미션 종료.", flush=True)
    else: print("⚠️ 도킹 실패.", flush=True)

    safety_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
