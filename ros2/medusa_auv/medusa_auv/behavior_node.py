import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Bool, Float32
from medusa_msgs.msg import JellyfishDetection, PropulsionState, AUVMissionCommand


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class BehaviorNode(Node):
    def __init__(self):
        super().__init__("behavior_node")
        self.observe_threshold = CFG.get("observe_threshold", 0.9)
        self.patience = CFG.get("behavior_patience", 20)
        self.relay_enabled = CFG.get("relay_enabled", True)
        self.dive_duration = CFG.get("dive_duration_s", 120.0)
        self.surface_duration = CFG.get("surface_duration_s", 20.0)
        self.battery_low_v = CFG.get("battery_low_v", 6.6)
        self.teleop_forward = CFG.get("teleop_forward_threshold", 0.3)
        self.teleop_back = CFG.get("teleop_back_threshold", -0.3)
        self.teleop_timeout = CFG.get("teleop_timeout_s", 2.0)
        self.mode = None
        self.lost = 0
        self.surfacing = False
        self.phase_elapsed = 0.0
        self.low_battery = False
        self.leaking = False
        self.manual_until = 0

        self.create_subscription(JellyfishDetection, "/auv/detection", self.on_detection, 10)
        self.create_subscription(Float32, "/auv/battery", self.on_battery, 10)
        self.create_subscription(Bool, "/auv/leak", self.on_leak, 10)
        self.create_subscription(Float32, "/auv/teleop", self.on_teleop, 10)
        self.mode_pub = self.create_publisher(UInt8, "/auv/behavior/mode", 10)
        self.mission_pub = self.create_publisher(AUVMissionCommand, "/auv/mission", 10)
        self.window_pub = self.create_publisher(Bool, "/auv/transmit_window", 10)

        self.set_mode(PropulsionState.SEARCH, AUVMissionCommand.SEARCH)
        self.publish_window(False)
        self.create_timer(1.0, self.relay_tick)

    def on_detection(self, msg):
        if self.surfacing or self.emergency() or self.manual_active():
            return
        if msg.jellyfish_detected:
            self.lost = 0
            if msg.confidence >= self.observe_threshold:
                self.set_mode(PropulsionState.OBSERVE, AUVMissionCommand.ACTIVATE)
            else:
                self.set_mode(PropulsionState.APPROACH, AUVMissionCommand.INTERCEPT)
        else:
            self.lost += 1
            if self.lost >= self.patience:
                self.set_mode(PropulsionState.SEARCH, AUVMissionCommand.SEARCH)

    def on_battery(self, msg):
        if self.low_battery or msg.data <= 0.0 or msg.data >= self.battery_low_v:
            return
        self.low_battery = True
        self.set_mode(PropulsionState.SURFACE, AUVMissionCommand.RETURN)
        self.publish_window(True)
        self.get_logger().warn(f"low battery {msg.data:.2f}V, surfacing")

    def on_leak(self, msg):
        if self.leaking or not msg.data:
            return
        self.leaking = True
        self.set_mode(PropulsionState.SURFACE, AUVMissionCommand.RETURN)
        self.publish_window(True)
        self.get_logger().error("leak detected, surfacing")

    def emergency(self):
        return self.low_battery or self.leaking

    def manual_active(self):
        return self.get_clock().now().nanoseconds < self.manual_until

    def on_teleop(self, msg):
        if self.emergency():
            return
        self.manual_until = self.get_clock().now().nanoseconds + int(self.teleop_timeout * 1e9)
        if msg.data >= self.teleop_forward:
            self.set_mode(PropulsionState.APPROACH, AUVMissionCommand.INTERCEPT)
        elif msg.data <= self.teleop_back:
            self.set_mode(PropulsionState.OBSERVE, AUVMissionCommand.ACTIVATE)
        else:
            self.set_mode(PropulsionState.SEARCH, AUVMissionCommand.SEARCH)

    def relay_tick(self):
        if not self.relay_enabled or self.emergency() or self.manual_active():
            return
        self.phase_elapsed += 1.0
        if self.surfacing:
            if self.phase_elapsed >= self.surface_duration:
                self.surfacing = False
                self.phase_elapsed = 0.0
                self.publish_window(False)
                self.set_mode(PropulsionState.SEARCH, AUVMissionCommand.SEARCH)
        elif self.phase_elapsed >= self.dive_duration:
            self.surfacing = True
            self.phase_elapsed = 0.0
            self.set_mode(PropulsionState.SURFACE, AUVMissionCommand.RETURN)
            self.publish_window(True)

    def publish_window(self, is_open):
        msg = Bool()
        msg.data = is_open
        self.window_pub.publish(msg)

    def set_mode(self, mode, command):
        if mode == self.mode:
            return
        self.mode = mode

        m = UInt8()
        m.data = mode
        self.mode_pub.publish(m)

        cmd = AUVMissionCommand()
        cmd.command = command
        self.mission_pub.publish(cmd)

        self.get_logger().info(f"mode={mode} command={command}")


def main():
    rclpy.init()
    node = BehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
