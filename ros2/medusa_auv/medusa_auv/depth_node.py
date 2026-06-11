import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, UInt8
from medusa_msgs.msg import PropulsionState

try:
    import ms5837
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        self.max_depth = CFG.get("operating_depth_max_m", 5.0)
        self.search_depth = CFG.get("search_depth_m", 1.5)
        self.ascent_rate = CFG.get("ascent_rate_mps", 0.1)
        self.descent_rate = CFG.get("descent_rate_mps", 0.05)
        self.rate = CFG.get("depth_update_hz", 5.0)
        self.depth = 0.0
        self.mode = PropulsionState.SEARCH

        self.sensor = None
        if SENSOR_AVAILABLE:
            self.sensor = ms5837.MS5837_30BA()
            self.sensor.init()

        self.create_subscription(UInt8, "/auv/behavior/mode", self.on_mode, 10)
        self.publisher = self.create_publisher(Float32, "/auv/depth", 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def on_mode(self, msg):
        self.mode = msg.data

    def tick(self):
        if self.sensor is not None and self.sensor.read():
            self.depth = self.sensor.depth()
        else:
            dt = 1.0 / self.rate
            if self.mode == PropulsionState.SURFACE:
                self.depth -= self.ascent_rate * dt
            elif self.depth < self.search_depth:
                self.depth += self.descent_rate * dt
            else:
                self.depth -= self.descent_rate * dt
            self.depth = max(0.0, min(self.max_depth, self.depth))

        msg = Float32()
        msg.data = float(self.depth)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = DepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
