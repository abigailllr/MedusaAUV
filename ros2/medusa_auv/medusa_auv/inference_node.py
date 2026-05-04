import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from medusa_msgs.msg import JellyfishDetection
import numpy as np
import tflite_runtime.interpreter as tflite
import yaml
import os


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        self.bridge = CvBridge()
        self.frame_id = 0
        self.threshold = CFG["inference_threshold"]
        self.img_size = CFG["img_size"]

        self.interpreter = tflite.Interpreter(model_path=CFG["model_path"])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.subscription = self.create_subscription(
            Image, "/auv/camera/image_raw", self.on_image, 10
        )
        self.publisher = self.create_publisher(JellyfishDetection, "/auv/detection", 10)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        import cv2
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        confidence = float(self.interpreter.get_tensor(self.output_details[0]["index"])[0][0])

        detection = JellyfishDetection()
        detection.header.stamp = self.get_clock().now().to_msg()
        detection.confidence = confidence
        detection.jellyfish_detected = confidence >= self.threshold
        detection.frame_id = self.frame_id
        self.frame_id += 1

        self.publisher.publish(detection)

        self.get_logger().info(
            f"confidence={confidence:.3f} detected={detection.jellyfish_detected}"
        )


def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
