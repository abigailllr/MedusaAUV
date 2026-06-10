import math
import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, UInt8
from medusa_msgs.msg import PropulsionState

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False

try:
    from adafruit_servokit import ServoKit
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class PropulsionNode(Node):
    def __init__(self):
        super().__init__("propulsion_node")
        self.pin = CFG["bell_gpio_pin"]
        self.channel = CFG.get("bell_pca_channel", 0)
        self.neutral = CFG["bell_neutral_deg"]
        self.amplitude = CFG["bell_amplitude_deg"]
        self.frequency = CFG["bell_pulse_init_hz"]
        self.min_hz = CFG["bell_pulse_min_hz"]
        self.max_hz = CFG["bell_pulse_max_hz"]
        self.surface_hz = CFG.get("surface_pulse_hz", CFG["bell_pulse_max_hz"])
        self.idle_w = CFG["actuator_idle_w"]
        self.stroke_j = CFG["actuator_stroke_j"]
        self.rate = CFG["control_rate_hz"]
        self.mode = PropulsionState.SEARCH
        self.phase = 0.0

        self.kit = None
        self.pwm = None
        if PCA_AVAILABLE:
            self.kit = ServoKit(channels=16)
        elif RPI_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pin, CFG["pwm_frequency"])
            self.pwm.start(0)

        self.create_subscription(Float32, "/auv/propulsion/freq_cmd", self.on_freq, 10)
        self.create_subscription(UInt8, "/auv/behavior/mode", self.on_mode, 10)
        self.state_pub = self.create_publisher(PropulsionState, "/auv/propulsion/state", 10)

        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(0.2, self.publish_state)

    def on_freq(self, msg):
        self.frequency = max(self.min_hz, min(self.max_hz, msg.data))

    def on_mode(self, msg):
        self.mode = msg.data

    def mode_amplitude(self):
        if self.mode == PropulsionState.OBSERVE:
            return self.amplitude * 0.4
        if self.mode == PropulsionState.IDLE:
            return 0.0
        return self.amplitude

    def effective_frequency(self):
        if self.mode == PropulsionState.SURFACE:
            return self.surface_hz
        return self.frequency

    def tick(self):
        self.phase += 2.0 * math.pi * self.effective_frequency() / self.rate
        if self.phase > 2.0 * math.pi:
            self.phase -= 2.0 * math.pi
        amp = self.mode_amplitude()
        angle = self.neutral + 0.5 * amp * (1.0 - math.cos(self.phase))
        self.apply_angle(angle)

    def apply_angle(self, angle):
        angle = max(0.0, min(180.0, angle))
        if self.kit is not None:
            self.kit.servo[self.channel].angle = angle
        elif self.pwm is not None:
            self.pwm.ChangeDutyCycle(2.5 + (angle / 180.0) * 10.0)

    def current_power(self):
        amp_norm = self.mode_amplitude() / max(self.amplitude, 1e-6)
        return self.idle_w + self.stroke_j * self.effective_frequency() * amp_norm

    def publish_state(self):
        state = PropulsionState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.pulse_frequency = float(self.effective_frequency())
        state.amplitude = float(self.mode_amplitude())
        state.power_w = float(self.current_power())
        state.mode = self.mode
        self.state_pub.publish(state)

    def destroy_node(self):
        if self.pwm is not None:
            self.pwm.stop()
            GPIO.cleanup()
        super().destroy_node()


def main():
    rclpy.init()
    node = PropulsionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
