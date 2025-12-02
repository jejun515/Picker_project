import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import math

class OrderManagerNode(Node):
    def __init__(self):
        super().__init__('order_manager_node')
        
        # 📡 토픽 발행 설정
        # 토픽명: /box_order_goals
        # 메시지 타입: geometry_msgs/PoseArray
        self.publisher_ = self.create_publisher(PoseArray, '/box_order_goals', 10)
        
        # 📦 박스 사이즈별 목표 좌표 (Map 좌표계 기준)
        # 예시: 창고의 선반 위치 등
        self.locations = {
            "Small":  {"x": -6.38, "y": 0.3, "z": 0.0},
            "Medium": {"x": -6.35, "y": 1.0, "z": 0.0},
            "Large":  {"x": -6.11, "y": 1.61, "z": 0.0},
            "XLarge": {"x": -6.12, "y": 2.23, "z:": 0.0}
        }
        
        self.get_logger().info('📦 Order Manager Node Started. Ready to publish goals.')

    def publish_goal(self, size: str):
        """
        주문받은 사이즈에 해당하는 좌표를 PoseArray에 담아 발행합니다.
        """
        if size not in self.locations:
            self.get_logger().warn(f'❌ Unknown size: {size}')
            return

        target = self.locations[size]
        
        # 1. PoseArray 메시지 생성
        msg = PoseArray()
        
        # 2. Header 설정 (필수: frame_id)
        msg.header = Header()
        msg.header.frame_id = "map"  # 지도 좌표계 기준
        msg.header.stamp = self.get_clock().now().to_msg()

        # 3. Pose 생성 (목표 위치)
        pose = Pose()
        pose.position.x = target["x"]
        pose.position.y = target["y"]
        pose.position.z = target["z"]
        
        # 방향(Orientation) 설정 - 예: 정면(0도) 바라보기
        # 쿼터니언 (x=0, y=0, z=0, w=1)
        pose.orientation.w = 1.0 
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0

        # 4. 리스트에 추가 (PoseArray는 여러 점을 보낼 수 있음)
        # 여기서는 주문 1건이므로 1개의 Pose만 담아서 보냄
        msg.poses.append(pose)

        # 5. 발행
        self.publisher_.publish(msg)
        self.get_logger().info(f'🚀 Published Goal for {size} box: (x={target["x"]}, y={target["y"]})')