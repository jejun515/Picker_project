import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from sensor_msgs.msg import Image


class YOLOMultiSubscriber(Node):
    def __init__(self):
        super().__init__('cctv_multi_subscriber')
        
        # CvBridge 객체 생성: 이미지 메시지 변환에 사용
        self.bridge = CvBridge()
        
        # ----------------------------------------------------
        # 1. 이미지 토픽 구독 설정 (화면 표시용)
        # ----------------------------------------------------
        self.image_subscription = self.create_subscription(
            Image,
            'cctvcam/image_processed',  # 이미지 토픽 이름
            self.image_callback,
            10
        )
        self.get_logger().info('✅ Image Topic 구독 시작: /cctvcam/image_processed')
        
        # ----------------------------------------------------
        # 2. Bool 상태 토픽 구독 설정 (상태 로깅용)
        # ----------------------------------------------------
        self.bool_subscription = self.create_subscription(
            Bool,
            'cctvcam/roi_status',       # Bool 토픽 이름
            self.bool_callback,
            10
        )
        self.get_logger().info('✅ Bool Topic 구독 시작: /cctvcam/roi_status')
        
        # 변수 저장 공간 (선택 사항: 두 토픽의 최신 상태를 저장할 때 유용)
        self.current_roi_status = False

    def image_callback(self, msg):
        """이미지 토픽을 수신하고 OpenCV 화면에 표시합니다."""
        try:
            # 🚀 [수정] desired_encoding='bgr8' 대신 'passthrough' 또는 'auto' 사용
            # 'passthrough'는 메시지에 명시된 인코딩을 사용하도록 CvBridge에 지시합니다.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # (선택 사항) 현재 Bool 상태를 이미지에 표시하여 시각적으로 확인
            status_text = "Object Detected!" if self.current_roi_status else "Clear"
            status_color = (0, 255, 0) if self.current_roi_status else (0, 0, 255)
            
            cv2.putText(cv_image, f"ROI Status: {status_text}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # OpenCV 창에 실시간으로 이미지 표시
            cv2.imshow("Processed CCTV Feed (Multi-Sub)", cv_image)
            
            # 키 입력 처리
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("q pressed. Shutting down...")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def bool_callback(self, msg):
        """Bool 토픽을 수신하고 상태를 로깅합니다."""
        roi_detected = msg.data 
        
        # 현재 상태를 변수에 저장 (image_callback에서 사용하기 위해)
        self.current_roi_status = roi_detected
        
        if roi_detected:
            self.get_logger().info('▶️ ROI STATUS: TRUE (객체 감지)')
        else:
            self.get_logger().info('◀️ ROI STATUS: FALSE (영역 비어있음)')


    def destroy_node(self):
        # 노드가 종료될 때 모든 OpenCV 창을 닫습니다.
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOMultiSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == '__main__':
    main()