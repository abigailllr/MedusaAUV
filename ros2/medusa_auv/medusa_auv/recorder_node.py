import os
import json
import yaml
import rclpy
from rclpy.node import Node
from medusa_msgs.msg import JellyfishDetection, PropulsionState, SwimMetrics, JellyfishSighting, AUVMissionCommand


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class RecorderNode(Node):
    def __init__(self):
        super().__init__("recorder_node")
        session_dir = os.path.join(
            os.path.dirname(__file__), "../../../", CFG.get("record_dir", "data/sessions")
        )
        os.makedirs(session_dir, exist_ok=True)
        stamp = self.get_clock().now().nanoseconds
        self.path = os.path.join(session_dir, f"session_{stamp}.jsonl")
        self.file = open(self.path, "a")

        self.create_subscription(JellyfishDetection, "/auv/detection", self.on_detection, 10)
        self.create_subscription(PropulsionState, "/auv/propulsion/state", self.on_propulsion, 10)
        self.create_subscription(SwimMetrics, "/auv/swim_metrics", self.on_metrics, 10)
        self.create_subscription(JellyfishSighting, "/auv/sighting", self.on_sighting, 10)
        self.create_subscription(AUVMissionCommand, "/auv/mission", self.on_mission, 10)

        self.get_logger().info(f"recording to {self.path}")

    def on_mission(self, msg):
        self.write("mission", {"command": int(msg.command)})

    def write(self, topic, data):
        record = {"t": self.get_clock().now().nanoseconds, "topic": topic, "data": data}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def on_detection(self, msg):
        self.write("detection", {
            "confidence": float(msg.confidence),
            "detected": bool(msg.jellyfish_detected),
            "frame_id": int(msg.frame_id),
        })

    def on_propulsion(self, msg):
        self.write("propulsion", {
            "frequency": float(msg.pulse_frequency),
            "power_w": float(msg.power_w),
            "mode": int(msg.mode),
        })

    def on_metrics(self, msg):
        self.write("metrics", {
            "speed": float(msg.estimated_speed),
            "body_lengths_per_sec": float(msg.body_lengths_per_sec),
            "cost_of_transport": float(msg.cost_of_transport),
        })

    def on_sighting(self, msg):
        self.write("sighting", {
            "lat": float(msg.latitude),
            "lon": float(msg.longitude),
            "density": float(msg.density_estimate),
            "severity": int(msg.bloom_severity),
        })

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = RecorderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
