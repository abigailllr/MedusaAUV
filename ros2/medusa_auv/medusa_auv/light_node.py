import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from medusa_msgs.msg import PropulsionState

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class LightNode(Node):
    def __init__(self):
        super().__init__("light_node")
        self.pin = CFG.get("lure_gpio_pin", 24)
        self.lure_modes = set(CFG.get("lure_modes", [PropulsionState.OBSERVE]))

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, False)

        self.create_subscription(UInt8, "/auv/behavior/mode", self.on_mode, 10)

    def on_mode(self, msg):
        on = msg.data in self.lure_modes
        if GPIO_AVAILABLE:
            GPIO.output(self.pin, on)
        self.get_logger().info(f"lure light {'on' if on else 'off'}")

    def destroy_node(self):
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        super().destroy_node()


def main():
    rclpy.init()
    node = LightNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
