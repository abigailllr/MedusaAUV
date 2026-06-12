import os
import json
import asyncio
import threading
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Bool, Float32
from medusa_msgs.msg import JellyfishDetection, PropulsionState, SwimMetrics, JellyfishSighting, MissionStatus
import websockets


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


class VRBridgeNode(Node):
    def __init__(self):
        super().__init__("vr_bridge_node")
        self.host = CFG.get("vr_bridge_host", "0.0.0.0")
        self.port = CFG.get("vr_bridge_port", 8765)
        self.rate = CFG.get("vr_bridge_rate_hz", 10.0)
        self.lock = threading.Lock()
        self.clients = set()
        self.state = {
            "confidence": 0.0,
            "jellyfish_detected": False,
            "pulse_frequency": 0.0,
            "power_w": 0.0,
            "mode": 0,
            "estimated_speed": 0.0,
            "body_lengths_per_sec": 0.0,
            "cost_of_transport": 0.0,
            "latitude": 0.0,
            "longitude": 0.0,
            "density_estimate": 0.0,
            "bloom_severity": 0,
            "surfaced": False,
            "depth": 0.0,
            "temperature": 0.0,
            "light": 0.0,
            "bloom_forecast": 0.0,
            "leak": False,
            "pulse_confidence": 0.0,
            "waypoint_distance": 0.0,
        }

        self.create_subscription(JellyfishDetection, "/auv/detection", self.on_detection, 10)
        self.create_subscription(PropulsionState, "/auv/propulsion/state", self.on_propulsion, 10)
        self.create_subscription(SwimMetrics, "/auv/swim_metrics", self.on_metrics, 10)
        self.create_subscription(JellyfishSighting, "/auv/sighting", self.on_sighting, 10)
        self.create_subscription(Bool, "/auv/transmit_window", self.on_window, 10)
        self.create_subscription(Float32, "/auv/depth", self.on_depth, 10)
        self.create_subscription(Float32, "/auv/temperature", self.on_temperature, 10)
        self.create_subscription(Float32, "/auv/light", self.on_light, 10)
        self.create_subscription(Float32, "/auv/bloom_forecast", self.on_forecast, 10)
        self.create_subscription(Bool, "/auv/leak", self.on_leak, 10)
        self.create_subscription(Float32, "/auv/pulse_confidence", self.on_pulse, 10)
        self.create_subscription(MissionStatus, "/auv/mission_status", self.on_mission, 10)
        self.mode_pub = self.create_publisher(UInt8, "/auv/behavior/mode", 10)
        self.teleop_pub = self.create_publisher(Float32, "/auv/teleop", 10)
        self.steer_pub = self.create_publisher(Float32, "/auv/steer", 10)

        threading.Thread(target=self.run_server, daemon=True).start()
        self.get_logger().info(f"vr bridge on ws://{self.host}:{self.port}")

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def on_detection(self, msg):
        self.update(confidence=float(msg.confidence), jellyfish_detected=bool(msg.jellyfish_detected))

    def on_propulsion(self, msg):
        self.update(pulse_frequency=float(msg.pulse_frequency), power_w=float(msg.power_w), mode=int(msg.mode))

    def on_metrics(self, msg):
        self.update(
            estimated_speed=float(msg.estimated_speed),
            body_lengths_per_sec=float(msg.body_lengths_per_sec),
            cost_of_transport=float(msg.cost_of_transport),
        )

    def on_sighting(self, msg):
        self.update(
            latitude=float(msg.latitude),
            longitude=float(msg.longitude),
            density_estimate=float(msg.density_estimate),
            bloom_severity=int(msg.bloom_severity),
        )

    def on_window(self, msg):
        self.update(surfaced=bool(msg.data))

    def on_depth(self, msg):
        self.update(depth=float(msg.data))

    def on_temperature(self, msg):
        self.update(temperature=float(msg.data))

    def on_light(self, msg):
        self.update(light=float(msg.data))

    def on_forecast(self, msg):
        self.update(bloom_forecast=float(msg.data))

    def on_leak(self, msg):
        self.update(leak=bool(msg.data))

    def on_pulse(self, msg):
        self.update(pulse_confidence=float(msg.data))

    def on_mission(self, msg):
        self.update(waypoint_distance=float(msg.distance_m))

    def snapshot(self):
        with self.lock:
            return json.dumps(self.state)

    def on_client_message(self, message):
        try:
            data = json.loads(message)
        except ValueError:
            return
        if "mode" in data:
            msg = UInt8()
            msg.data = int(data["mode"])
            self.mode_pub.publish(msg)
        if "control" in data:
            msg = Float32()
            msg.data = float(data["control"])
            self.teleop_pub.publish(msg)
        if "steer" in data:
            msg = Float32()
            msg.data = float(data["steer"])
            self.steer_pub.publish(msg)

    async def handler(self, ws, path=None):
        self.clients.add(ws)
        try:
            async for message in ws:
                self.on_client_message(message)
        finally:
            self.clients.discard(ws)

    async def broadcast(self):
        while True:
            if self.clients:
                websockets.broadcast(self.clients, self.snapshot())
            await asyncio.sleep(1.0 / self.rate)

    async def serve(self):
        await websockets.serve(self.handler, self.host, self.port)
        await self.broadcast()

    def run_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.serve())


def main():
    rclpy.init()
    node = VRBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
