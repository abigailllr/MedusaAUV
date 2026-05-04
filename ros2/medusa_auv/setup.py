from setuptools import setup

package_name = "medusa_auv"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "camera_node    = medusa_auv.camera_node:main",
            "inference_node = medusa_auv.inference_node:main",
            "servo_node     = medusa_auv.servo_node:main",
            "telemetry_node = medusa_auv.telemetry_node:main",
        ],
    },
)
