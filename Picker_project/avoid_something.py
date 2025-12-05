import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Int32, Bool 
from irobot_create_msgs.msg import AudioNoteVector, AudioNote 
from builtin_interfaces.msg import Duration 
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
# 1. 안전 가드 + 통신(Phase + ROI) 모듈
# =========================================
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        ns = self.get_namespace()
        prefix = "" if ns == "/" else ns

        self.scan_sub = self.create_subscription(LaserScan, f'{prefix}/scan', self.scan_callback, qos)
        self.input_sub = self.create_subscription(Twist, '/cmd_vel_input', self.input_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'{prefix}/cmd_vel', 10)
        self.img_sub = self.create_subscription(Image, f'{prefix}/oakd/rgb/preview/image_raw', self.img_callback, qos)
        self.order_sub = self.create_subscription(PoseArray, f'{prefix}/box_order_goals', self.order_callback, 10)
        
        self.count_pub = self.create_publisher(Int32, '/camera/box_count', 10)
        self.audio_pub = self.create_publisher(AudioNoteVector, f'{prefix}/cmd_audio', 10)

        # [통신 1] Phase 통신
        self.phase_pub = self.create_publisher(Int32, f'{prefix}/current_phase', 10)
        target_ns = '/robot3' if 'robot2' in ns else '/robot2'
        self.other_robot_phase = 0 
        self.phase_sub = self.create_subscription(Int32, f'{target_ns}/current_phase', self.phase_callback, 10)
        
        # [통신 2] ROI 통신
        self.roi_pub = self.create_publisher(Bool, '/ROI_robot_detected', 10)
        self.roi_sub = self.create_subscription(Bool, '/ROI_robot_detected', self.roi_callback, 10)
        
        self.is_roi_occupied = False      
        self.am_i_working = False         
        self.last_roi_free_time = 0.0

        self.bridge = CvBridge()
        self.latest_cv_image = None
        
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
        self.received_poses = []
        self.has_new_order = False

    def phase_callback(self, msg):
        self.other_robot_phase = msg.data

    def roi_callback(self, msg):
        if self.am_i_working: return 
        if msg.data:
            self.is_roi_occupied = True
        else:
            if self.is_roi_occupied:
                self.last_roi_free_time = time.time()
            self.is_roi_occupied = False

    def publish_phase(self, phase_num: int):
        msg = Int32()
        msg.data = phase_num
        self.phase_pub.publish(msg)

    def set_roi_status(self, status: bool):
        self.am_i_working = status
        msg = Bool()
        msg.data = status
        self.roi_pub.publish(msg)
        state_str = "작업 중(True)" if status else "작업 끝(False)"
        print(f"📡 [ROI] 상태 변경 -> {state_str}", flush=True)

    def play_arrival_sound(self):
        print("🎵 [SOUND] 목적지 도착! 알림음 재생 (3초)", flush=True)
        for i in range(10):
            msg = AudioNoteVector()
            if i % 2 == 0: freq = 880
            else: freq = 440
            note = AudioNote(frequency=freq, max_runtime=Duration(sec=0, nanosec=300000000))
            msg.notes.append(note)
            msg.append = False
            self.audio_pub.publish(msg)
            time.sleep(0.3)

    def scan_callback(self, msg):
        self.is_sensor_active = True
        ranges = msg.ranges
        count = len(ranges)
        if count == 0: return

        CENTER_RATIO = 0.25 
        center_idx = int(count * CENTER_RATIO)
        stop_fov = 50 / 360
        stop_width = int(count * stop_fov / 2)
        s_start = max(0, center_idx - stop_width)
        s_end = min(count, center_idx + stop_width)
        stop_ranges = ranges[s_start : s_end]
        valid_stop = [r for r in stop_ranges if 0.18 < r < 1.0]
        min_dist = min(valid_stop) if valid_stop else 10.0
        self.current_dist = min_dist
        self.is_danger = (min_dist < self.emergency_dist)

        steer_fov = 100 / 360
        steer_width = int(count * steer_fov / 2)
        w_start = max(0, center_idx - steer_width)
        w_end = min(count, center_idx + steer_width)
        wide_ranges = ranges[w_start : w_end]
        mid = len(wide_ranges) // 2
        valid_l = [r for r in wide_ranges[:mid] if r > 0.18]
        valid_r = [r for r in wide_ranges[mid:] if r > 0.18]
        l_avg = sum(valid_l) / len(valid_l) if valid_l else 0.0
        r_avg = sum(valid_r) / len(valid_r) if valid_r else 0.0
        if l_avg > r_avg: self.obstacle_dir = 1.0 
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

    def order_callback(self, msg):
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
    
    # [1] 주문 대기
    safety_node.publish_phase(0)
    print("\n🌐 [대기 중] '/box_order_goals' 토픽을 기다리는 중...", flush=True)
    while not safety_node.has_new_order:
        safety_node.publish_phase(0)
        time.sleep(1.0)     
    
    if not navigator.getDockedStatus(): navigator.dock()
    # initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    # navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    navigator.undock()
    
    print("⏳ 센서 확인 중...", flush=True)
    while not safety_node.is_sensor_active: time.sleep(0.1)
    print("✅ 센서 연결됨.", flush=True)

    # -------------------------------------------------------------
    # 목표 설정
    # -------------------------------------------------------------
    box_pose_raw = safety_node.received_poses[0]
    room_pose_raw = safety_node.received_poses[1]
    
    target_box_x = box_pose_raw.position.x
    target_box_y = box_pose_raw.position.y
    target_room_x = room_pose_raw.position.x
    target_room_y = room_pose_raw.position.y
    
    # Phase 5 복귀 좌표 및 도착 방향
    ns = safety_node.get_namespace()
    dock_prep_x = 0.0
    dock_prep_y = 0.0

    if 'robot3' in ns:
        dock_prep_x = 0.12
        dock_prep_y = 0.65
        print("🤖 Robot 3 설정 적용.", flush=True)
    elif 'robot2' in ns:
        dock_prep_x = -0.4
        dock_prep_y = 0.5
        print("🤖 Robot 2 설정 적용.", flush=True)
    else:
        print("⚠️ Unknown Robot Namespace. 기본값 사용.")

    print(f"🚀 미션 시작! 1차목표: ({target_box_x}, {target_box_y})", flush=True)

    prefix = "" if ns == "/" else ns
    config_cli = safety_node.create_client(SetParameters, f'{prefix}/controller_server/set_parameters')
    
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
        
        time.sleep(1.5) 
        
        print("⏳ 경로 계산 중...", flush=True)
        wait_start = time.time()
        path_valid = False
        
        while time.time() - wait_start < 5.0:
            feedback = navigator.getFeedback()
            if feedback and feedback.distance_remaining > arrival_radius:
                path_valid = True
                print(f"✅ 경로 확보됨 (남은 거리: {feedback.distance_remaining:.2f}m)", flush=True)
                break
            time.sleep(0.1)

        last_known_dist = float('inf')
        start_time = time.time()

        while not navigator.isTaskComplete():
            if safety_node.is_danger:
                print(f"🚨 [장애물] {safety_node.current_dist:.2f}m -> 회피!", flush=True)
                navigator.cancelTask()
                stop_twist = Twist(); stop_twist.linear.x = -0.15
                safety_node.cmd_vel_pub.publish(stop_twist); time.sleep(0.5)
                
                print("🔄 회피 중...", flush=True)
                while safety_node.is_danger:
                    twist = Twist(); twist.linear.x = 0.0
                    twist.angular.z = 2.5 * safety_node.obstacle_dir 
                    safety_node.cmd_vel_pub.publish(twist)
                    time.sleep(0.1)
                
                print("✅ 탈출 성공. 재출발.", flush=True)
                go_twist = Twist(); go_twist.linear.x = 0.2
                safety_node.cmd_vel_pub.publish(go_twist); time.sleep(0.5)
                safety_node.cmd_vel_pub.publish(Twist()); time.sleep(0.1)
                return "RETRY"

            feedback = navigator.getFeedback()
            if feedback:
                dist = feedback.distance_remaining
                last_known_dist = dist
                if not strict_mode and dist < arrival_radius:
                    if path_valid and (time.time() - start_time < 3.0):
                        continue
                    print(f"🚩 [도착] 반경 진입 ({dist:.2f}m).", flush=True)
                    navigator.cancelTask(); safety_node.cmd_vel_pub.publish(Twist())
                    return "SUCCESS"
            time.sleep(0.05)

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED: return "SUCCESS"
        elif result == TaskResult.CANCELED: return "RETRY"
        limit = arrival_radius + 0.05 if strict_mode else arrival_radius + 0.3
        return "SUCCESS" if last_known_dist < limit else "FAIL"

    def nudge_robot(distance_m, speed_mps=0.05, safety_limit=None):
        action = "전진" if distance_m > 0 else "후진"
        print(f"📏 [마무리] {abs(distance_m)}m {action}...", flush=True)
        duration = abs(distance_m) / speed_mps
        twist = Twist(); twist.linear.x = speed_mps if distance_m > 0 else -speed_mps
        start_time = time.time()
        while (time.time() - start_time) < duration:
            if distance_m > 0 and safety_limit is not None:
                if safety_node.current_dist < safety_limit:
                    print(f"🛑 [STOP] 장애물 근접! ({safety_node.current_dist:.2f}m)", flush=True)
                    break
            safety_node.cmd_vel_pub.publish(twist); time.sleep(0.1)
        safety_node.cmd_vel_pub.publish(Twist())

    # =========================================================
    # [NEW] 스마트 교통 정리 (Phase + ROI 체크 + 10초 대기)
    # =========================================================
    def check_and_wait_if_busy():
        if 'robot2' in ns:
            other_phase = safety_node.other_robot_phase
            is_roi = safety_node.is_roi_occupied
            
            # (Phase가 1,2,3) OR (ROI가 True)이면 위험!
            if (other_phase in [1, 2, 3]) or is_roi:
                print(f"\n⚠️ [TRAFFIC] 작업 중 감지! (Phase: {other_phase}, ROI: {is_roi})", flush=True)
                print("🛑 [WAIT] 대기 장소로 이동합니다.", flush=True)
                
                hold_pose = navigator.getPoseStamped([-0.4, -3.35], TurtleBot4Directions.EAST)
                drive_smart(hold_pose, arrival_radius=0.2, strict_mode=False)
                
                print("⏳ 안전거리 확보를 위해 수초간 대기...", flush=True)
                time.sleep(25.0)
                
                print("👀 [CHECK] 진입 조건 확인 중...", flush=True)
                while True:
                    curr_phase = safety_node.other_robot_phase
                    curr_roi = safety_node.is_roi_occupied
                    
                    if (curr_phase == 0 or curr_phase >= 4) and (not curr_roi):
                        print(f"\n✅ [GO] 진입 허가! (Phase: {curr_phase}, ROI: False)", flush=True)
                        break
                    
                    print(f"⏳ 대기 중... (Phase: {curr_phase}, ROI: {curr_roi})", end='\r', flush=True)
                    time.sleep(1.0)

    # =========================================================
    # Phase 1: 1차 진입
    # =========================================================
    check_and_wait_if_busy()

    safety_node.publish_phase(1)
    safety_node.emergency_dist = 0.50 
    goal_1 = navigator.getPoseStamped([2.0, -4.0], TurtleBot4Directions.NORTH)
    set_nav2_params(0.31, 0.5, 3.14)
    
    while True:
        status = drive_smart(goal_1, arrival_radius=1.0, strict_mode=False)
        if status == "SUCCESS": print("✅ 1차 진입 완료.", flush=True); break
        elif status == "RETRY": continue
        else: print("❌ 1차 실패.", flush=True); rclpy.shutdown(); return

    # =========================================================
    # Phase 2: 박스 위치
    # =========================================================
    check_and_wait_if_busy()

    safety_node.publish_phase(2)
    print("\n📉 [접근] 안전거리 15cm로 축소.", flush=True)
    safety_node.emergency_dist = 0.15 
    
    goal_2 = navigator.getPoseStamped([target_box_x, target_box_y], TurtleBot4Directions.NORTH_EAST)
    set_nav2_params(0.1, 0.05, 0.1)
    
    while True:
        status = drive_smart(goal_2, arrival_radius=0.05, strict_mode=True)
        if status == "SUCCESS": 
            print("🎉 박스 앞 도착!", flush=True)
            nudge_robot(0.15, safety_limit=0.40) 
            break
        elif status == "RETRY": continue
        else: print("❌ 도착 실패.", flush=True); rclpy.shutdown(); return

    # =========================================================
    # Phase 3: YOLO (ROI 점유)
    # =========================================================
    safety_node.publish_phase(3)
    safety_node.set_roi_status(True) # Bool True 발행
    
    print("\n=== [Phase 3] 물체 감지 시작 ===", flush=True)
    time.sleep(2.0)
    box_count = safety_node.detect_and_count()
    print(f"\n📦📦📦 [결과] 감지된 박스 개수: {box_count} 개 📦📦📦\n", flush=True)

    print("🔙 후진하여 거리 확보.", flush=True)
    nudge_robot(-0.25)
    print("📈 [복구] 안전거리 0.5m로 복구.", flush=True)
    safety_node.emergency_dist = 0.50
    
    safety_node.set_roi_status(False) # Bool False 발행

    # =========================================================
    # Phase 4: 도착지로 이동
    # =========================================================
    safety_node.publish_phase(4) # Phase 4 알림 (뒷 로봇 진입 허용)
    print("\n=== [Phase 4] 도착지로 이동 ===", flush=True)
    goal_3 = navigator.getPoseStamped([target_room_x, target_room_y], TurtleBot4Directions.NORTH)
    set_nav2_params(0.31, 0.2, 0.5) 

    while True:
        status = drive_smart(goal_3, arrival_radius=0.2, strict_mode=False)
        if status == "SUCCESS": 
            print("✅ 2차 지점 도착 완료!", flush=True)
            safety_node.play_arrival_sound()
            
            print("\n📡 [DATA] DB 업데이트를 위해 박스 개수 전송 중...", flush=True)
            final_count = max(0, box_count)
            msg = Int32()
            msg.data = final_count
            safety_node.count_pub.publish(msg)
            
            print(f"   -> 전송 개수: {final_count}개 (/camera/box_count)\n", flush=True)
            break
            
        elif status == "RETRY": continue
        else: print("❌ 이동 실패.", flush=True); rclpy.shutdown(); return
    
    time.sleep(1.0)

    # =========================================================
    # Phase 5: 도킹 복귀
    # =========================================================
    safety_node.publish_phase(5)
    print("\n=== [Phase 5] 도킹 스테이션 복귀 ===", flush=True)
    print(f"📍 도킹 준비 위치: ({dock_prep_x}, {dock_prep_y})")
    
    dock_pose = navigator.getPoseStamped([dock_prep_x, dock_prep_y], TurtleBot4Directions.SOUTH)
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
