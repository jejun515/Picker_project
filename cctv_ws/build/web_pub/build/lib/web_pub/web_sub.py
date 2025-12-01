import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class YOLOWebcamSublisher(Node):
    def __init__(self):
        super().__init__('webcam_subscriber')
        self.subscription = self.create_subscription(
            Bool,
            'webcam_msg',
            self.data_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        self.get_logger().info('Data Subscriber Node has been started.')

    def data_callback(self, msg):
        # Parse the received message, expecting the format: "id,timestamp"
        sp = msg.data
        if sp:
            self.get_logger().info('msg success')
        else:
            self.get_logger().info('msg failed')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOWebcamSublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()