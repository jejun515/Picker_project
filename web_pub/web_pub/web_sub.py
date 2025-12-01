import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class YOLOWebcamSublisher(Node):
    def __init__(self):
        super().__init__('cctvcam_subscriber')
        self.subscription = self.create_subscription(
            Bool,
            'cctvcam_msg',
            self.data_callback,
            10
        )

        self.subscription_ = self.create_subscription(
            CompressedImage,
            'camera_image',
            self.image_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        self.get_logger().info('Data Subscriber Node has been started.')
        self.bridge = CvBridge()

    def data_callback(self, msg):
        # Parse the received message, expecting the format: "id,timestamp"
        sp = msg.data
        if sp:
            self.get_logger().info('msg success')
        else:
            self.get_logger().info('msg failed')

    def image_callback(self, msg):
        self.get_logger().info('Received image')
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Display the image using OpenCV
        cv2.imshow('Received Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YOLOWebcamSublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()  # Close OpenCV windows
        rclpy.shutdown()
