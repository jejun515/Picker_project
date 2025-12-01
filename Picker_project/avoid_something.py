import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav2_simple_commander.robot_navigator import TaskResult
import time
import math

# =========================================
# 1. 안전 가드 & 중재자 클래스 (Phase 1, 2 공용)
# =========================================
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        qos = QoSProfile(depth=10)
        
        # 센서 구독
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        
        # 팀원 명령 구독 (Phase 2용)
        self.input_sub = self.create_subscription(Twist, '/cmd_vel_input', self.input_callback, qos)
        
        # 로봇 제어 (긴급 회피 및 중개용)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        # 설정값
        self.emergency_dist = 0.40  # 이 거리보다 가까우면 비상 상황
        self.is_danger = False
        self.phase2_active = False # Phase 2 시작 전엔 팀원 명령 무시

    def scan_callback(self, msg):
        # 전방 50도(-25 ~ +25) 감시
        ranges = msg.ranges
        # Turtlebot4의 LIDAR 데이터 배열 구조에 따라 슬라이싱
        front_ranges = ranges[0:45] + ranges[-45:]
        
        min_dist = float('inf')
        for r in front_ranges:
            if 0.1 < r < self.emergency_dist: # 노이즈(0.1) 제외
                if r < min_dist: min_dist = r
        
        self.is_danger = (min_dist < self.emergency_dist)

    def input_callback(self, msg):
        # Phase 2가 아니면 무시
        if not self.phase2_active:
            return

        final_cmd = Twist()
        
        if self.is_danger:
            # [Phase 2] 장애물 감지 시: 팀원 명령 무시하고 제자리 회피
            # (로그 너무 많이 뜨지 않게 throttle 조절 필요할 수 있음)
            # print("🚨 [Phase 2] 장애물 감지! 회피 기동 중...")
            final_cmd.linear.x = 0.0
            final_cmd.angular.z = 0.5 # 왼쪽으로 회전
        else:
            # [Phase 2] 안전함: 팀원 명령 통과
            final_cmd = msg
            
        self.cmd_vel_pub.publish(final_cmd)

    def execute_manual_evasion(self):
        # [Phase 1] Nav2 주행 중 긴급 회피 동작
        print("⚡ [Phase 1] 긴급 회피 발동! (Nav2 잠시 비켜!)")
        twist = Twist()
        
        # 정지 -> 후진 -> 회전
        twist.linear.x = 0.0; self.cmd_vel_pub.publish(twist); time.sleep(0.2)
        twist.linear.x = -0.15; self.cmd_vel_pub.publish(twist); time.sleep(0.5)
        twist.linear.x = 0.0; twist.angular.z = 0.8; self.cmd_vel_pub.publish(twist); time.sleep(1.0)
        
        # 정지
        twist.angular.z = 0.0; self.cmd_vel_pub.publish(twist)

# =========================================
# 2. Nav2 설정 변경 클래스
# =========================================
class Nav2Configurator(Node):
    def __init__(self):
        super().__init__('nav2_configurator')
        self.cli = self.create_client(SetParameters, '/robot3/controller_server/set_parameters')

    def set_params(self, max_speed, xy_tol, yaw_tol):
        if not self.cli.wait_for_service(timeout_sec=1.0):
            print("⚠️ Controller Server 연결 실패. 기본 설정으로 주행합니다.")
            return

        req = SetParameters.Request()
        # DWB Controller 파라미터 이름 (로봇 설정에 따라 다를 수 있음)
        req.parameters = [
            Parameter(name='FollowPath.max_vel_x', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=max_speed)),
            Parameter(name='FollowPath.xy_goal_tolerance', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=xy_tol)),
            Parameter(name='FollowPath.yaw_goal_tolerance', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=yaw_tol))
        ]
        self.cli.call_async(req)
        print(f"🔧 Nav2 설정 변경: 속도={max_speed}, 거리오차={xy_tol}, 각도오차={yaw_tol}")
        time.sleep(0.5) # 적용 대기

# =========================================
# 3. 메인 실행 로직
# =========================================
def main():
    rclpy.init()
    
    # 노드 생성
    safety_node = SafetyMonitor()
    config_node = Nav2Configurator()
    navigator = TurtleBot4Navigator()

    # --- 초기화 ---
    if not navigator.getDockedStatus():
        navigator.info('Checking Dock Status...')
        navigator.dock()

    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    navigator.undock()

    # ---------------------------------------------------------
    # 함수: Nav2 이동 + 장애물 감시 + 강제 성공 처리
    # ---------------------------------------------------------
    def drive_smart(target_pose, arrival_radius):
        print(f"🚗 이동 시작! (목표 반경 {arrival_radius}m 진입 시 성공 처리)")
        navigator.startToPose(target_pose)
        
        last_known_dist = float('inf')

        while not navigator.isTaskComplete():
            # 1. 거리 체크 및 강제 성공 판정
            feedback = navigator.getFeedback()
            if feedback:
                dist = feedback.distance_remaining
                last_known_dist = dist
                
                if dist < arrival_radius:
                    print(f"🚩 [이동 중] 목표 반경 진입 ({dist:.2f}m). 정지합니다.")
                    navigator.cancelTask()
                    safety_node.cmd_vel_pub.publish(Twist()) # 정지
                    return "SUCCESS"

            # 2. 장애물 감시
            rclpy.spin_once(safety_node, timeout_sec=0.05)
            if safety_node.is_danger:
                print("🚨 장애물 감지! Nav2 중단 및 회피!")
                navigator.cancelTask()
                safety_node.execute_manual_evasion()
                return "RETRY"

        # --- Nav2 종료 후 결과 확인 (여기가 수정됨) ---
        result = navigator.getResult()
        print(f"🧐 Nav2 결과 코드(원본): {result}") 

        # TaskResult 객체와 직접 비교해야 정확합니다.
        if result == TaskResult.SUCCEEDED:
            return "SUCCESS"
        elif result == TaskResult.CANCELED:
            return "RETRY"
        elif result == TaskResult.FAILED:
            # 실패했지만 거리가 가까우면 성공 처리
            if last_known_dist < arrival_radius + 0.3:
                print(f"⚠️ Nav2는 실패(FAILED)라지만, 목표 근처({last_known_dist:.2f}m)입니다. [성공 처리]")
                return "SUCCESS"
            else:
                return "FAIL"
        else:
            return "FAIL"

    # =========================================================
    # Phase 1-1: 중간 지점 이동
    # =========================================================
    # 좌표는 질문주신 로그에 맞춰 수정했습니다 (-4.5, 0.4)
    goal_1 = navigator.getPoseStamped([-4.5, 0.4], TurtleBot4Directions.SOUTH)
    
    # 속도 0.31, xy오차 1.0 (넓게), 각도오차 3.14 (무시)
    config_node.set_params(max_speed=0.31, xy_tol=1.0, yaw_tol=3.14)
    
    while True:
        status = drive_smart(goal_1, arrival_radius=1.0)
        
        # 디버깅: 함수가 뭘 리턴했는지 눈으로 확인
        print(f"👉 drive_smart 리턴값: {status}") 

        if status == "SUCCESS": # <--- 문자열 비교
            print("✅ 1차 목표 통과.")
            break
        elif status == "RETRY":
            print("🔄 경로 재설정 중...")
            continue
        else:
            print("❌ 1차 이동 실패. 프로그램을 종료합니다.")
            rclpy.shutdown()
            return

    # =========================================================
    # Phase 1-1: 중간 지점 이동 (빠르게, 대충)
    # =========================================================
    goal_1 = navigator.getPoseStamped([-4.5, 0.4], TurtleBot4Directions.SOUTH)
    
    # 속도 0.31(최대), 도착 반경 1.0m로 설정 (제자리 회전 방지용으로 Yaw 오차 크게)
    config_node.set_params(max_speed=0.31, xy_tol=1.0, yaw_tol=3.14)
    
    while True:
        # 반경 1.0m 안에만 들면 성공으로 침
        status = drive_smart(goal_1, arrival_radius=1.0)
        
        if status == "SUCCESS":
            print("✅ 1차 목표 통과.")
            break
        elif status == "RETRY":
            print("🔄 경로 재설정 중...")
            continue
        else:
            print("❌ 1차 이동 실패. 프로그램을 종료합니다.")
            rclpy.shutdown()
            return

    # =========================================================
    # Phase 1-2: 최종 지점 이동 (느리게, 정확하게)
    # =========================================================
    goal_2 = navigator.getPoseStamped([-6.4, 0.28], TurtleBot4Directions.SOUTH)
    
    # 속도 0.15(저속), 도착 반경 0.1m(정밀)
    config_node.set_params(max_speed=0.15, xy_tol=0.1, yaw_tol=3.14)
    
    while True:
        # 반경 0.3m 안에 들면 성공으로 침 (너무 좁게 잡으면 못 멈춤)
        status = drive_smart(goal_2, arrival_radius=0.1)
        
        if status == "SUCCESS":
            print("🎉 최종 목표 도착 완료!")
            break
        elif status == "RETRY":
            continue
        else:
            print("❌ 최종 이동 실패.")
            rclpy.shutdown()
            return

    # =========================================================
    # Phase 2: 물체 추적 모드 (팀원 코드 연동)
    # =========================================================
    print("\n=== [Phase 2] 추적 모드 전환 ===")
    print("👉 팀원에게 알리세요: '/cmd_vel_input' 토픽으로 명령을 보내주세요.")
    
    safety_node.phase2_active = True # 이제부터 SafetyMonitor가 중재 시작
    
    try:
        # SafetyMonitor가 계속 돌면서 중재 역할 수행
        while rclpy.ok():
            rclpy.spin_once(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        print("프로그램 종료.")
        safety_node.destroy_node()
        config_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
