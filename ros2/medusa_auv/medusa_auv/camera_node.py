import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.publisher = self.create_publisher(Image, "/auv/camera/image_raw", 10)
        self.bridge = CvBridge()
        self.width = CFG.get("camera_width", 640)
        self.height = CFG.get("camera_height", 480)

        source = CFG.get("camera_source", "picamera")
        self.picam = None
        self.cap = None
        if source == "picamera" and PICAMERA_AVAILABLE:
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            ))
            self.picam.start()
        else:
            device = 0 if source == "picamera" else source
            self.cap = cv2.VideoCapture(device)
            if not self.cap.isOpened():
                self.get_logger().error("camera source could not be opened")

        fps = CFG.get("camera_fps", 20)
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

    def publish_frame(self):
        if self.picam is not None:
            frame = self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("no camera frame", throttle_duration_sec=5.0)
                return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam is not None:
            self.picam.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
