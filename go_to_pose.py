import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from cv_bridge import CvBridge

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image 

from tf_transformations import quaternion_from_euler
import cv2
from ultralytics import YOLO
import math


DETECT_TARGET = [0]  # 탐지할 객체 클래스 ID (0: blue_car, 1: green_car)

QUADRANT_TARGET_POSES = {
    1: (0.00464523, 3.32853, math.pi / 2),
    2: (-2.37742, 4.07556, 3 * math.pi / 4),
    3: (-4.30382, 1.58606, -3 * math.pi / 4),
    4: (-1.51404, -0.197009, -math.pi / 2)
}


class MoveRobot(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.get_logger().info('[INFO] MoveRobot 노드가 실행되었습니다.')

        self.model = YOLO('./mixed_results.pt')

        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("[ERROR] 웹캠을 열 수 없습니다.")
            raise IOError("Webcam failed to open.")

        self.bridge = CvBridge() 
        self.image_publisher = self.create_publisher(Image, '/webcam/annotated_frame', 10)
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        self.previous_quadrant = -1
        self.timer = self.create_timer(0.5, self.detection_loop)

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

        self.get_logger().info(f'[INFO] 고객의 위치가 변경되었습니다.')
        self._action_client.send_goal_async(goal_msg)

    def detection_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("[WARN] 카메라 프레임을 읽을 수 없습니다.")
            return

        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        results = self.model.predict(frame, classes=DETECT_TARGET, conf=0.5, verbose=False)
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
                cv2.putText(annotated_frame, f'Q{new_quadrant}', (xc, yc + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break 

        # 사분면 변경 감지 및 목표 전송
        if new_quadrant != -1 and new_quadrant != self.previous_quadrant:
            self.get_logger().info(f"[INFO] 고객 이동 감지: {self.previous_quadrant}사분면 -> {new_quadrant}사분면")
            
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
            self.timer.cancel()
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        yolo_node = MoveRobot()
        rclpy.spin(yolo_node)
    except IOError:
        pass 
    except Exception as e:
        rclpy.get_logger('main').error(f"[ERROR] Main logger: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()