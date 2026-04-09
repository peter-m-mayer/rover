"""
PyBullet-based digital twin simulator for the RASPBOT-V2.

Provides simulated Camera, Sensors, and Actuators that plug into the
SLAM pipeline identically to the real hardware interfaces. The simulator
renders a 3D indoor environment with textured walls, furniture, and a
mecanum-wheeled rover.

Usage:
    from raspbot_slam.simulator import SimWorld, SimCamera, SimSensors, SimActuators

    world = SimWorld(floor_plan="L_shaped")
    camera = SimCamera(world)
    sensors = SimSensors(world)
    actuators = SimActuators(world)

    # These drop into the SLAM pipeline in place of real hardware:
    frame = camera.capture()
    range_mm = sensors.read_ultrasonic_mm()
    actuators.move_forward(60)
"""

from .sim_world import SimWorld
from .sim_camera import SimCamera
from .sim_sensors import SimSensors
from .sim_actuators import SimActuators
from .sim_imu import SimIMU
