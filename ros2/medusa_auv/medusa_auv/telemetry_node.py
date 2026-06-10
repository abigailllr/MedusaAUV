import rclpy
from rclpy.node import Node
from medusa_msgs.msg import JellyfishDetection, PropulsionState, SwimMetrics
import rerun as rr


class TelemetryNode(Node):
    def __init__(self):
        super().__init__("telemetry_node")
        rr.init("medusa_auv", spawn=True)

        self.create_subscription(JellyfishDetection, "/auv/detection", self.on_detection, 10)
        self.create_subscription(PropulsionState, "/auv/propulsion/state", self.on_propulsion, 10)
        self.create_subscription(SwimMetrics, "/auv/swim_metrics", self.on_metrics, 10)

    def on_detection(self, msg):
        rr.log("auv/detection/confidence", rr.Scalar(msg.confidence))
        rr.log("auv/detection/jellyfish", rr.Scalar(float(msg.jellyfish_detected)))

    def on_propulsion(self, msg):
        rr.log("auv/propulsion/frequency", rr.Scalar(msg.pulse_frequency))
        rr.log("auv/propulsion/power_w", rr.Scalar(msg.power_w))
        rr.log("auv/propulsion/mode", rr.Scalar(float(msg.mode)))

    def on_metrics(self, msg):
        rr.log("auv/metrics/estimated_speed", rr.Scalar(msg.estimated_speed))
        rr.log("auv/metrics/body_lengths_per_sec", rr.Scalar(msg.body_lengths_per_sec))
        rr.log("auv/metrics/cost_of_transport", rr.Scalar(msg.cost_of_transport))


def main():
    rclpy.init()
    node = TelemetryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
