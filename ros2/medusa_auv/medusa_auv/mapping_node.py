import os
import json
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from medusa_msgs.msg import JellyfishDetection, JellyfishSighting
from medusa_auv.algorithms import bloom_severity


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class MappingNode(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.cell = CFG["map_cell_deg"]
        self.low = CFG["bloom_low_count"]
        self.high = CFG["bloom_high_count"]
        self.out_path = os.path.join(
            os.path.dirname(__file__), "../../../", CFG["map_output_path"]
        )
        self.grid = {}
        self.last_fix = None

        self.create_subscription(NavSatFix, "/auv/gps", self.on_gps, 10)
        self.create_subscription(JellyfishDetection, "/auv/detection", self.on_detection, 10)
        self.sighting_pub = self.create_publisher(JellyfishSighting, "/auv/sighting", 10)

        self.create_timer(CFG.get("map_save_interval_s", 10.0), self.save_map)

    def on_gps(self, msg):
        self.last_fix = (msg.latitude, msg.longitude)

    def key(self, lat, lon):
        return (round(lat / self.cell), round(lon / self.cell))

    def on_detection(self, msg):
        if not msg.jellyfish_detected or self.last_fix is None:
            return
        lat, lon = self.last_fix
        cell = self.grid.setdefault(
            self.key(lat, lon), {"count": 0, "conf_sum": 0.0, "lat": lat, "lon": lon}
        )
        cell["count"] += 1
        cell["conf_sum"] += msg.confidence
        cell["lat"], cell["lon"] = lat, lon

        sighting = JellyfishSighting()
        sighting.header.stamp = self.get_clock().now().to_msg()
        sighting.latitude = float(lat)
        sighting.longitude = float(lon)
        sighting.confidence = float(msg.confidence)
        sighting.density_estimate = float(cell["count"])
        sighting.bloom_severity = bloom_severity(cell["count"], self.low, self.high)
        self.sighting_pub.publish(sighting)

    def save_map(self):
        features = []
        for cell in self.grid.values():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [cell["lon"], cell["lat"]]},
                "properties": {
                    "count": cell["count"],
                    "mean_confidence": cell["conf_sum"] / max(cell["count"], 1),
                    "bloom_severity": bloom_severity(cell["count"], self.low, self.high),
                },
            })
        feature_collection = {"type": "FeatureCollection", "features": features}
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(feature_collection, f, indent=2)
        self.get_logger().info(f"wrote {len(features)} cells to {self.out_path}")


def main():
    rclpy.init()
    node = MappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
