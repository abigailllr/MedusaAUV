import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from medusa_msgs.msg import JellyfishSighting
from medusa_auv.algorithms import bloom_forecast


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class PredictionNode(Node):
    def __init__(self):
        super().__init__("prediction_node")
        self.opt_min = CFG.get("optimal_temp_min_c", 25.0)
        self.opt_max = CFG.get("optimal_temp_max_c", 27.0)
        self.falloff = CFG.get("temp_falloff_c", 5.0)
        self.density_reference = CFG.get("density_reference", 10.0)
        self.rate = CFG.get("forecast_update_hz", 1.0)

        self.temperature = 0.0
        self.light = 0.0
        self.density = 0.0

        self.create_subscription(Float32, "/auv/temperature", self.on_temp, 10)
        self.create_subscription(Float32, "/auv/light", self.on_light, 10)
        self.create_subscription(JellyfishSighting, "/auv/sighting", self.on_sighting, 10)
        self.publisher = self.create_publisher(Float32, "/auv/bloom_forecast", 10)
        self.create_timer(1.0 / self.rate, self.publish_forecast)

    def on_temp(self, msg):
        self.temperature = msg.data

    def on_light(self, msg):
        self.light = msg.data

    def on_sighting(self, msg):
        self.density = msg.density_estimate

    def publish_forecast(self):
        forecast = bloom_forecast(
            self.temperature, self.light, self.density,
            self.opt_min, self.opt_max, self.falloff, self.density_reference,
        )
        msg = Float32()
        msg.data = float(forecast)
        self.publisher.publish(msg)
        self.get_logger().info(f"bloom_forecast={msg.data:.2f} temp={self.temperature:.1f}C light={self.light:.2f}")


def main():
    rclpy.init()
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
