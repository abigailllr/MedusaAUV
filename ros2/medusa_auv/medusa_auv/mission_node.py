import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from medusa_msgs.msg import MissionStatus
from medusa_auv.algorithms import haversine, bearing


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")
        self.waypoints = CFG.get("waypoints", [])
        self.arrival_radius = CFG.get("waypoint_arrival_m", 3.0)
        self.rate = CFG.get("mission_update_hz", 1.0)
        self.index = 0
        self.fix = None

        self.create_subscription(NavSatFix, "/auv/gps", self.on_gps, 10)
        self.publisher = self.create_publisher(MissionStatus, "/auv/mission_status", 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def on_gps(self, msg):
        self.fix = (msg.latitude, msg.longitude)

    def tick(self):
        if not self.waypoints or self.fix is None:
            return
        target = self.waypoints[self.index]
        distance = haversine(self.fix, target)
        arrived = distance <= self.arrival_radius

        msg = MissionStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.distance_m = float(distance)
        msg.bearing_deg = float(bearing(self.fix, target))
        msg.arrived = arrived
        msg.waypoint_index = self.index
        self.publisher.publish(msg)

        if arrived:
            self.index = (self.index + 1) % len(self.waypoints)


def main():
    rclpy.init()
    node = MissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
