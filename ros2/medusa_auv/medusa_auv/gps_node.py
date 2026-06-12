import os
import math
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

try:
    import serial
    import pynmea2
    GPS_SERIAL_AVAILABLE = True
except ImportError:
    GPS_SERIAL_AVAILABLE = False


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

        self.serial = None
        if GPS_SERIAL_AVAILABLE:
            try:
                self.serial = serial.Serial(
                    CFG.get("gps_serial_port", "/dev/ttyUSB2"),
                    CFG.get("gps_baud", 115200),
                    timeout=1.0,
                )
            except serial.SerialException:
                self.get_logger().warn("gps serial unavailable, using simulation")

        self.publisher = self.create_publisher(NavSatFix, "/auv/gps", 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def read_serial(self):
        try:
            line = self.serial.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("$") and ("GGA" in line or "RMC" in line):
                parsed = pynmea2.parse(line)
                if parsed.latitude and parsed.longitude:
                    return float(parsed.latitude), float(parsed.longitude)
        except (pynmea2.ParseError, serial.SerialException, ValueError):
            return None
        return None

    def tick(self):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        fix = self.read_serial() if self.serial is not None else None
        if fix is not None:
            msg.latitude, msg.longitude = fix
        else:
            t = (self.get_clock().now().nanoseconds - self.start) / 1e9
            msg.latitude = self.home_lat + self.wander * math.sin(t * 0.05)
            msg.longitude = self.home_lon + self.wander * math.cos(t * 0.05)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = GpsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
