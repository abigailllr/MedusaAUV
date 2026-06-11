import os
import math
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from medusa_msgs.msg import PropulsionState

try:
    import board
    import adafruit_mpu6050
    IMU_AVAILABLE = True
except ImportError:
    IMU_AVAILABLE = False


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu_node")
        self.rate = CFG.get("imu_update_hz", 50.0)
        self.sim_natural = CFG.get("sim_natural_hz", 0.6)
        self.sim_bandwidth = CFG.get("sim_bandwidth_hz", 0.2)
        self.frequency = CFG.get("bell_pulse_init_hz", 0.5)
        self.phase = 0.0

        self.sensor = adafruit_mpu6050.MPU6050(board.I2C()) if IMU_AVAILABLE else None
        self.create_subscription(PropulsionState, "/auv/propulsion/state", self.on_propulsion, 10)
        self.publisher = self.create_publisher(Imu, "/auv/imu", 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def on_propulsion(self, msg):
        self.frequency = msg.pulse_frequency

    def tick(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.sensor is not None:
            ax, ay, az = self.sensor.acceleration
            msg.linear_acceleration.x = float(ax)
            msg.linear_acceleration.y = float(ay)
            msg.linear_acceleration.z = float(az)
        else:
            self.phase += 2.0 * math.pi * self.frequency / self.rate
            gain = math.exp(-((self.frequency - self.sim_natural) / self.sim_bandwidth) ** 2)
            msg.linear_acceleration.x = float(gain * math.sin(self.phase))
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = ImuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
