import os
import math
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class GpsNode(Node):
    def __init__(self):
        super().__init__("gps_node")
        self.home_lat = CFG.get("home_lat", 47.36)
        self.home_lon = CFG.get("home_lon", 8.54)
        self.wander = CFG.get("gps_wander_deg", 0.0002)
        self.rate = CFG.get("gps_update_hz", 1.0)
        self.start = self.get_clock().now().nanoseconds
        self.publisher = self.create_publisher(NavSatFix, "/auv/gps", 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def tick(self):
        t = (self.get_clock().now().nanoseconds - self.start) / 1e9
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.latitude = self.home_lat + self.wander * math.sin(t * 0.05)
        msg.longitude = self.home_lon + self.wander * math.cos(t * 0.05)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = GpsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
