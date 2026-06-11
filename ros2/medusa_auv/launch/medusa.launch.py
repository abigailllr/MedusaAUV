from launch import LaunchDescription
from launch_ros.actions import Node


NODES = [
    "camera_node",
    "inference_node",
    "propulsion_node",
    "resonance_tuner_node",
    "behavior_node",
    "mapping_node",
    "telemetry_node",
    "vr_bridge_node",
    "recorder_node",
    "battery_node",
    "depth_node",
    "gps_node",
    "environment_node",
    "prediction_node",
    "leak_node",
    "mission_node",
    "pulse_node",
    "light_node",
    "imu_node",
]


def generate_launch_description():
    nodes = [Node(package="medusa_auv", executable=name, name=name) for name in NODES]
    nodes.append(Node(package="web_video_server", executable="web_video_server", name="web_video_server"))
    return LaunchDescription(nodes)
