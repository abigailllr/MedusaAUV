import os
import math
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from w1thermsensor import W1ThermSensor
    TEMP_SENSOR_AVAILABLE = True
except ImportError:
    TEMP_SENSOR_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class EnvironmentNode(Node):
    def __init__(self):
        super().__init__("environment_node")
        self.base_temp = CFG.get("water_temp_c", 24.0)
        self.temp_swing = CFG.get("water_temp_swing_c", 3.0)
        self.day_cycle = CFG.get("day_cycle_s", 600.0)
        self.rate = CFG.get("environment_update_hz", 1.0)
        self.start = self.get_clock().now().nanoseconds

        self.sensor = W1ThermSensor() if TEMP_SENSOR_AVAILABLE else None
        self.temp_pub = self.create_publisher(Float32, "/auv/temperature", 10)
        self.light_pub = self.create_publisher(Float32, "/auv/light", 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def tick(self):
        t = (self.get_clock().now().nanoseconds - self.start) / 1e9
        phase = math.sin(2.0 * math.pi * t / self.day_cycle)

        if self.sensor is not None:
            temperature = self.sensor.get_temperature()
        else:
            temperature = self.base_temp + self.temp_swing * phase

        temp_msg = Float32()
        temp_msg.data = float(temperature)
        self.temp_pub.publish(temp_msg)

        light_msg = Float32()
        light_msg.data = float(max(0.0, phase))
        self.light_pub.publish(light_msg)


def main():
    rclpy.init()
    node = EnvironmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
