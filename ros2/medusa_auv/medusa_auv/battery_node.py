import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    import board
    import adafruit_ina219
    INA_AVAILABLE = True
except ImportError:
    INA_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class BatteryNode(Node):
    def __init__(self):
        super().__init__("battery_node")
        self.full_v = CFG.get("battery_full_v", 8.4)
        self.empty_v = CFG.get("battery_empty_v", 6.0)
        self.sim_seconds = CFG.get("sim_battery_minutes", 10.0) * 60.0
        self.interval = CFG.get("battery_check_interval_s", 2.0)
        self.start = self.get_clock().now().nanoseconds

        self.sensor = None
        if INA_AVAILABLE:
            self.sensor = adafruit_ina219.INA219(board.I2C())

        self.publisher = self.create_publisher(Float32, "/auv/battery", 10)
        self.create_timer(self.interval, self.publish_battery)

    def read_voltage(self):
        if self.sensor is not None:
            return float(self.sensor.bus_voltage)
        elapsed = (self.get_clock().now().nanoseconds - self.start) / 1e9
        fraction = min(elapsed / self.sim_seconds, 1.0)
        return self.full_v - (self.full_v - self.empty_v) * fraction

    def publish_battery(self):
        msg = Float32()
        msg.data = float(self.read_voltage())
        self.publisher.publish(msg)
        self.get_logger().info(f"battery={msg.data:.2f}V")


def main():
    rclpy.init()
    node = BatteryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
