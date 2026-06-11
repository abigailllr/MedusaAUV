import os
from glob import glob
from setuptools import setup

package_name = "medusa_auv"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    data_files=[
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    entry_points={
        "console_scripts": [
            "camera_node           = medusa_auv.camera_node:main",
            "inference_node        = medusa_auv.inference_node:main",
            "telemetry_node        = medusa_auv.telemetry_node:main",
            "propulsion_node       = medusa_auv.propulsion_node:main",
            "resonance_tuner_node  = medusa_auv.resonance_tuner_node:main",
            "behavior_node         = medusa_auv.behavior_node:main",
            "mapping_node          = medusa_auv.mapping_node:main",
            "vr_bridge_node        = medusa_auv.vr_bridge_node:main",
            "recorder_node         = medusa_auv.recorder_node:main",
            "battery_node          = medusa_auv.battery_node:main",
            "depth_node            = medusa_auv.depth_node:main",
            "gps_node              = medusa_auv.gps_node:main",
            "environment_node      = medusa_auv.environment_node:main",
            "prediction_node       = medusa_auv.prediction_node:main",
            "leak_node             = medusa_auv.leak_node:main",
            "mission_node          = medusa_auv.mission_node:main",
            "pulse_node            = medusa_auv.pulse_node:main",
            "light_node            = medusa_auv.light_node:main",
        ],
    },
)
