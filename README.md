# MedusaAUV

An autonomous underwater robot that swims with a resonance tuned silicone bell and runs a MobileNetV2 TFLite detector, fused with a pulse rhythm cue, on a Raspberry Pi for real time onboard jellyfish detection, streaming a live first person view to a WebXR VR headset over WiFi and 4G LTE, all on a ROS 2 stack.

## Hardware

- Raspberry Pi 5 and Camera Module 3
- 8x SG90 class servos on a PCA9685 I2C driver
- MPU6050 IMU, MS5837 depth sensor, DS18B20 temperature, leak sensor
- 4x 18650 UPS HAT and magnetic reed switch
- SIM7670G 4G LTE and GPS HAT
- Meta Quest 3
- Clear acrylic flotation hull and Ecoflex silicone bell
