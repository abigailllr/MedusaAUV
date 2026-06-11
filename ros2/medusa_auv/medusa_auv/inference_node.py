import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import UInt8, Float32
from medusa_msgs.msg import JellyfishDetection, PropulsionState
from medusa_auv.algorithms import gray_world, fuse_confidence
from collections import deque
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import yaml
import os


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        self.bridge = CvBridge()
        self.frame_id = 0
        self.threshold = CFG["inference_threshold"]
        self.img_size = CFG["img_size"]
        self.window = CFG.get("smoothing_window", 12)
        self.conf_history = deque(maxlen=self.window)
        self.stride = CFG.get("inference_stride", 1)
        self.stride_search = CFG.get("inference_stride_search", self.stride)
        self.mode = PropulsionState.SEARCH
        self.skip = 0
        self.pulse_confidence = 0.0
        self.pulse_weight = CFG.get("fusion_model_weight", 0.7)
        self.dataset_capture = CFG.get("dataset_capture", False)
        self.dataset_min = CFG.get("dataset_min_confidence", 0.8)
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "../../../", CFG.get("dataset_dir", "data/dataset"))

        self.interpreter = tflite.Interpreter(model_path=CFG["model_path"])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.subscription = self.create_subscription(
            Image, "/auv/camera/image_raw", self.on_image, 10
        )
        self.create_subscription(UInt8, "/auv/behavior/mode", self.on_mode, 10)
        self.create_subscription(Float32, "/auv/pulse_confidence", self.on_pulse, 10)
        self.publisher = self.create_publisher(JellyfishDetection, "/auv/detection", 10)

    def on_mode(self, msg):
        self.mode = msg.data

    def on_pulse(self, msg):
        self.pulse_confidence = msg.data

    def current_stride(self):
        if self.mode == PropulsionState.SEARCH:
            return max(self.stride_search, 1)
        return max(self.stride, 1)

    def capture_dataset(self, frame, confidence):
        if not self.dataset_capture or confidence < self.dataset_min:
            return
        os.makedirs(self.dataset_dir, exist_ok=True)
        path = os.path.join(self.dataset_dir, f"jelly_{self.frame_id}_{int(confidence * 100)}.jpg")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def on_image(self, msg):
        self.skip += 1
        if self.skip < self.current_stride():
            return
        self.skip = 0

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        balanced = gray_world(frame.astype(np.float32))
        resized = cv2.resize(balanced, (self.img_size, self.img_size))
        tensor = np.expand_dims(resized / 255.0, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        confidence = float(self.interpreter.get_tensor(self.output_details[0]["index"])[0][0])
        fused = fuse_confidence(confidence, self.pulse_confidence, self.pulse_weight)

        self.conf_history.append(fused)
        smoothed = float(np.mean(self.conf_history))

        detection = JellyfishDetection()
        detection.header.stamp = self.get_clock().now().to_msg()
        detection.confidence = smoothed
        detection.jellyfish_detected = smoothed >= self.threshold
        detection.frame_id = self.frame_id
        self.frame_id += 1

        self.publisher.publish(detection)
        self.capture_dataset(frame, smoothed)

        self.get_logger().info(
            f"model={confidence:.3f} fused={fused:.3f} smoothed={smoothed:.3f} detected={detection.jellyfish_detected}"
        )


def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
