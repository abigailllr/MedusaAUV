import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from collections import deque
from medusa_auv.algorithms import band_ratio
import numpy as np
import cv2


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class PulseNode(Node):
    def __init__(self):
        super().__init__("pulse_node")
        self.fps = CFG.get("camera_fps", 20)
        self.window = CFG.get("pulse_window_frames", 64)
        self.band_min = CFG.get("pulse_band_min_hz", 0.25)
        self.band_max = CFG.get("pulse_band_max_hz", 1.0)
        self.bridge = CvBridge()
        self.prev = None
        self.motion = deque(maxlen=self.window)

        self.create_subscription(Image, "/auv/camera/image_raw", self.on_image, 10)
        self.publisher = self.create_publisher(Float32, "/auv/pulse_confidence", 10)

    def on_image(self, msg):
        gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            self.motion.append(float(np.mean(cv2.absdiff(gray, self.prev))))
        self.prev = gray
        if len(self.motion) < self.window:
            return

        confidence = band_ratio(self.motion, self.fps, self.band_min, self.band_max)
        out = Float32()
        out.data = confidence
        self.publisher.publish(out)
        self.get_logger().info(f"pulse_confidence={confidence:.3f}")


def main():
    rclpy.init()
    node = PulseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
