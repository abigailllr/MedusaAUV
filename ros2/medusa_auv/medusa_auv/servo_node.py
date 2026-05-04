import rclpy
from rclpy.node import Node
from medusa_msgs.msg import JellyfishDetection, ServoState
import yaml
import os

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

SERVO_PIN = CFG["servo_gpio_pin"]
MIN_ANGLE = CFG["servo_min_angle"]
MAX_ANGLE = CFG["servo_max_angle"]


class ServoNode(Node):
    def __init__(self):
        super().__init__("servo_node")
        self.current_angle = 0.0
        self.pwm = None

        if RPI_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self.pwm = GPIO.PWM(SERVO_PIN, 50)
            self.pwm.start(0)

        self.subscription = self.create_subscription(
            JellyfishDetection, "/auv/detection", self.on_detection, 10
        )
        self.publisher = self.create_publisher(ServoState, "/auv/servo_state", 10)

    def angle_to_duty(self, angle):
        return 2.5 + (angle / 180.0) * 10.0

    def move_to(self, angle):
        angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        if self.pwm:
            self.pwm.ChangeDutyCycle(self.angle_to_duty(angle))
        self.current_angle = angle

    def on_detection(self, msg):
        target = MAX_ANGLE if msg.jellyfish_detected else MIN_ANGLE
        self.move_to(target)

        state = ServoState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.current_angle = self.current_angle
        state.target_angle = float(target)
        state.mechanism_active = msg.jellyfish_detected
        self.publisher.publish(state)

        self.get_logger().info(
            f"servo angle={self.current_angle} active={state.mechanism_active}"
        )

    def destroy_node(self):
        if self.pwm:
            self.pwm.stop()
            GPIO.cleanup()
        super().destroy_node()


def main():
    rclpy.init()
    node = ServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
