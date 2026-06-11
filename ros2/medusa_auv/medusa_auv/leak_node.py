import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class LeakNode(Node):
    def __init__(self):
        super().__init__("leak_node")
        self.pin = CFG.get("leak_gpio_pin", 23)
        self.interval = CFG.get("leak_check_interval_s", 1.0)

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        self.publisher = self.create_publisher(Bool, "/auv/leak", 10)
        self.create_timer(self.interval, self.tick)

    def tick(self):
        leak = bool(GPIO.input(self.pin)) if GPIO_AVAILABLE else False
        msg = Bool()
        msg.data = leak
        self.publisher.publish(msg)
        if leak:
            self.get_logger().error("leak detected")

    def destroy_node(self):
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        super().destroy_node()


def main():
    rclpy.init()
    node = LeakNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
