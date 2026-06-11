import math
import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from medusa_msgs.msg import PropulsionState, SwimMetrics


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class ResonanceTunerNode(Node):
    def __init__(self):
        super().__init__("resonance_tuner_node")
        self.freq = CFG["bell_pulse_init_hz"]
        self.min_hz = CFG["bell_pulse_min_hz"]
        self.max_hz = CFG["bell_pulse_max_hz"]
        self.step = CFG["tuner_step_hz"]
        self.body_length = CFG["body_length_m"]
        self.mass = CFG["vehicle_mass_kg"]
        self.imu_speed_gain = CFG.get("imu_speed_gain", 0.1)
        self.sim_natural = CFG.get("sim_natural_hz", 0.6)
        self.sim_bandwidth = CFG.get("sim_bandwidth_hz", 0.2)

        self.acc_mean = 0.0
        self.acc_ms = 0.0
        self.have_imu = False
        self.power = CFG["actuator_idle_w"]
        self.direction = 1.0
        self.last_obj = 0.0

        self.create_subscription(Imu, "/auv/imu", self.on_imu, 10)
        self.create_subscription(PropulsionState, "/auv/propulsion/state", self.on_state, 10)
        self.freq_pub = self.create_publisher(Float32, "/auv/propulsion/freq_cmd", 10)
        self.metrics_pub = self.create_publisher(SwimMetrics, "/auv/swim_metrics", 10)

        self.create_timer(0.5, self.publish_metrics)
        self.create_timer(CFG["tuner_interval_s"], self.tune_step)
        self.publish_freq()

    def on_imu(self, msg):
        self.have_imu = True
        ax = msg.linear_acceleration.x
        self.acc_mean = 0.99 * self.acc_mean + 0.01 * ax
        hp = ax - self.acc_mean
        self.acc_ms = 0.99 * self.acc_ms + 0.01 * hp * hp

    def on_state(self, msg):
        self.power = msg.power_w

    def objective(self):
        if self.have_imu:
            return math.sqrt(max(self.acc_ms, 0.0))
        return self.body_length * math.exp(-((self.freq - self.sim_natural) / self.sim_bandwidth) ** 2)

    def tune_step(self):
        j = self.objective()
        if j < self.last_obj:
            self.direction *= -1.0
        self.last_obj = j
        self.freq = self.freq + self.direction * self.step
        if self.freq <= self.min_hz:
            self.freq = self.min_hz
            self.direction = 1.0
        if self.freq >= self.max_hz:
            self.freq = self.max_hz
            self.direction = -1.0
        self.publish_freq()

    def publish_freq(self):
        msg = Float32()
        msg.data = float(self.freq)
        self.freq_pub.publish(msg)

    def publish_metrics(self):
        j = self.objective()
        speed = j * self.imu_speed_gain if self.have_imu else j
        cot = self.power / (self.mass * 9.81 * speed) if speed >= 1e-3 else -1.0
        msg = SwimMetrics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pulse_frequency = float(self.freq)
        msg.estimated_speed = float(speed)
        msg.body_lengths_per_sec = float(speed / max(self.body_length, 1e-6))
        msg.cost_of_transport = float(cot)
        msg.thrust_proxy = float(j)
        self.metrics_pub.publish(msg)
        self.get_logger().info(
            f"freq={self.freq:.3f}Hz speed={speed:.3f}m/s cot={cot:.3f}"
        )


def main():
    rclpy.init()
    node = ResonanceTunerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
