import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import math

class OrderManagerNode(Node):
    def __init__(self):
        super().__init__('order_manager_node')
        
        # 📡 토픽 발행 설정
        self.topic_name = '/box_order_goals'
        self.publisher_ = self.create_publisher(PoseArray, self.topic_name, 10)
        
        # 📦 목표 좌표 데이터베이스 (Map 좌표계 기준)
        self.locations = {
            # === [1번 목적지] 박스 재고 위치 (키 이름 변경됨) ===
            "S_box":  {"x": -6.38, "y": 0.3, "z": 0.0},
            "M_box":  {"x": -6.35, "y": 1.0, "z": 0.0},
            "L_box":  {"x": -6.11, "y": 1.61, "z": 0.0},
            "XL_box": {"x": -6.12, "y": 2.23, "z": 0.0},

            # === [2번 목적지] 탈의실 위치 ===
            "ChangingRoom1": {"x": -0.35, "y": 3.65, "z": 0.0}, # 탈의실 1번
            "ChangingRoom2": {"x": 0.17, "y": 3.54, "z": 0.0}   # 탈의실 2번
        }
        
        print(f"✅ [ROS] Order Manager Node Started. Publisher Topic: {self.topic_name}")

    def publish_goal(self, location_key: str):
        """
        입력받은 키(location_key)에 해당하는 좌표를 PoseArray에 담아 발행합니다.
        예: 'S_box', 'ChangingRoom1' 등
        """
        # 데이터 매핑 확인
        if location_key not in self.locations:
            print(f"❌ [ROS ERROR] Unknown location key received: {location_key}")
            # 가능한 키 목록을 보여주어 디버깅을 도움
            print(f"   -> Available keys: {list(self.locations.keys())}")
            return

        target = self.locations[location_key]
        
        try:
            # 1. PoseArray 메시지 생성
            msg = PoseArray()
            msg.header = Header()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()

            # 2. Pose 생성
            pose = Pose()
            pose.position.x = float(target["x"])
            pose.position.y = float(target["y"])
            pose.position.z = float(target["z"])
            
            # 방향 (정면)
            pose.orientation.w = 1.0 
            
            msg.poses.append(pose)

            # 3. 토픽 발행
            self.publisher_.publish(msg)

            # ✨ [디버그 출력]
            print("\n" + "="*40)
            print(f"📡 [ROS PUBLISH] Topic: {self.topic_name}")
            print(f"🏁 [DATA] Destination: {location_key}")
            print(f"📍 [DATA] Coordinates: X={target['x']}, Y={target['y']}")
            print("="*40 + "\n")
            
        except Exception as e:
            print(f"🔥 [ROS ERROR] Failed to publish goal: {e}")