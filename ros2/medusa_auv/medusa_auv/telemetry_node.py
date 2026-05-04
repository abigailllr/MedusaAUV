import rclpy
from rclpy.node import Node
from medusa_msgs.msg import JellyfishDetection, ServoState
import rerun as rr


class TelemetryNode(Node):
    def __init__(self):
        super().__init__("telemetry_node")
        rr.init("medusa_robotics", spawn=True)

        self.create_subscription(JellyfishDetection, "/auv/detection", self.on_detection, 10)
        self.create_subscription(ServoState, "/auv/servo_state", self.on_servo, 10)

    def on_detection(self, msg):
        rr.log("auv/detection/confidence", rr.Scalar(msg.confidence))
        rr.log("auv/detection/jellyfish", rr.Scalar(float(msg.jellyfish_detected)))

    def on_servo(self, msg):
        rr.log("auv/servo/angle", rr.Scalar(msg.current_angle))
        rr.log("auv/servo/active", rr.Scalar(float(msg.mechanism_active)))


def main():
    rclpy.init()
    node = TelemetryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
