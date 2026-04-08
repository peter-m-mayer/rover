"""
PyBullet world: physics simulation, room construction, and rover model.

Creates a 3D indoor environment with textured walls, floors, and simple
furniture. Spawns a mecanum rover with realistic dimensions and mass.
Steps the physics simulation and provides the ground-truth robot pose.
"""

import math
import os
from typing import Tuple, Optional, List, Dict
import numpy as np

import pybullet as p
import pybullet_data

try:
    import cv2
except ImportError:
    cv2 = None

from .. import config


# =============================================================================
# Floor Plan Definitions
# =============================================================================

# Wall segments: list of ((x1,y1), (x2,y2), height, thickness)
# Coordinates in meters, origin at room center

FLOOR_PLANS = {
    "simple_room": {
        "description": "4m x 4m single room",
        "walls": [
            ((-2, -2), (2, -2)),    # south
            ((2, -2), (2, 2)),      # east
            ((2, 2), (-2, 2)),      # north
            ((-2, 2), (-2, -2)),    # west
        ],
        "floor_size": (4.0, 4.0),
        "furniture": [
            {"type": "box", "pos": (1.2, 1.2, 0.3), "size": (0.4, 0.6, 0.3), "color": (0.6, 0.3, 0.1, 1)},
            {"type": "box", "pos": (-1.0, 0.5, 0.25), "size": (0.3, 0.3, 0.25), "color": (0.4, 0.4, 0.7, 1)},
        ],
    },
    "L_shaped": {
        "description": "L-shaped room (6m x 4m with 2m x 2m cutout)",
        "walls": [
            ((-3, -2), (3, -2)),     # south
            ((3, -2), (3, 0)),       # east lower
            ((3, 0), (1, 0)),        # east step
            ((1, 0), (1, 2)),        # east upper
            ((1, 2), (-3, 2)),       # north
            ((-3, 2), (-3, -2)),     # west
        ],
        "floor_size": (6.0, 4.0),
        "furniture": [
            {"type": "box", "pos": (-1.5, -0.5, 0.35), "size": (0.8, 0.4, 0.35), "color": (0.5, 0.3, 0.2, 1)},
            {"type": "box", "pos": (0.5, 1.0, 0.2), "size": (0.3, 0.3, 0.2), "color": (0.3, 0.5, 0.3, 1)},
            {"type": "cylinder", "pos": (-2.0, 1.0, 0.5), "radius": 0.15, "height": 0.5, "color": (0.7, 0.7, 0.2, 1)},
        ],
    },
    "corridor": {
        "description": "6m long, 1.5m wide corridor",
        "walls": [
            ((-3, -0.75), (3, -0.75)),
            ((3, -0.75), (3, 0.75)),
            ((3, 0.75), (-3, 0.75)),
            ((-3, 0.75), (-3, -0.75)),
        ],
        "floor_size": (6.0, 1.5),
        "furniture": [],
    },
    "two_rooms": {
        "description": "Two 3m x 3m rooms connected by a 1m doorway",
        "walls": [
            # Room 1
            ((-3, -1.5), (0, -1.5)),   # south
            ((0, -1.5), (0, -0.5)),    # divider south
            ((0, 0.5), (0, 1.5)),      # divider north (doorway gap -0.5 to 0.5)
            ((-3, 1.5), (-3, -1.5)),   # west
            ((-3, 1.5), (0, 1.5)),     # north room 1
            # Room 2
            ((0, -1.5), (3, -1.5)),    # south
            ((3, -1.5), (3, 1.5)),     # east
            ((3, 1.5), (0, 1.5)),      # north room 2
        ],
        "floor_size": (6.0, 3.0),
        "furniture": [
            {"type": "box", "pos": (-1.5, 0, 0.3), "size": (0.5, 0.5, 0.3), "color": (0.6, 0.2, 0.2, 1)},
            {"type": "box", "pos": (1.5, 0.5, 0.4), "size": (0.6, 0.3, 0.4), "color": (0.2, 0.2, 0.6, 1)},
        ],
    },
}


class SimWorld:
    """PyBullet simulation world with indoor environment and rover.

    Usage:
        world = SimWorld(floor_plan="L_shaped", gui=True)
        world.step()  # advance physics
        pose = world.get_robot_pose()  # ground truth
        world.close()
    """

    WALL_HEIGHT = 1.5       # meters
    WALL_THICKNESS = 0.05   # meters
    ROVER_HEIGHT = 0.08     # rover chassis height above floor
    ROVER_SIZE = (0.18, 0.16, 0.06)  # length, width, height (meters)
    ROVER_MASS = 1.5        # kg

    def __init__(self, floor_plan: str = "simple_room", gui: bool = False,
                 start_pose: Tuple[float, float, float] = (0, 0, 0),
                 time_step: float = 1/240.0):
        """Initialize the simulation.

        Args:
            floor_plan: Name of floor plan from FLOOR_PLANS dict.
            gui: If True, open PyBullet GUI window.
            start_pose: (x, y, theta) starting pose for the rover.
            time_step: Physics timestep in seconds.
        """
        self.floor_plan_name = floor_plan
        self.plan = FLOOR_PLANS.get(floor_plan, FLOOR_PLANS["simple_room"])
        self._gui = gui
        self._time_step = time_step
        self._start_pose = start_pose

        # Connect to PyBullet
        if gui:
            self._physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # Build the world
        self._body_ids: List[int] = []
        self._wall_ids: List[int] = []
        self._build_floor()
        self._build_walls()
        self._build_furniture()
        self._rover_id = self._build_rover(start_pose)

        # Texture management
        self._wall_textures_applied = False
        self._apply_textures()

        # Motor state
        self._motor_velocities = [0.0, 0.0, 0.0, 0.0]  # L1, L2, R1, R2

        # Servo state
        self._pan_angle = config.SERVO_PAN_CENTER
        self._tilt_angle = config.SERVO_TILT_REST

        # Step counter
        self._step_count = 0

    def step(self, n: int = 1):
        """Advance simulation by n steps."""
        for _ in range(n):
            self._apply_motor_forces()
            p.stepSimulation()
            self._step_count += 1

    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Get ground-truth robot pose (x, y, theta).

        Returns:
            (x, y, theta) where theta is yaw in radians, CCW from +X.
        """
        pos, orn = p.getBasePositionAndOrientation(self._rover_id)
        euler = p.getEulerFromQuaternion(orn)
        return (pos[0], pos[1], euler[2])

    def get_robot_position_3d(self) -> Tuple[float, float, float]:
        """Full 3D position (x, y, z)."""
        pos, _ = p.getBasePositionAndOrientation(self._rover_id)
        return pos

    def get_robot_orientation(self):
        """Get quaternion orientation."""
        _, orn = p.getBasePositionAndOrientation(self._rover_id)
        return orn

    def get_robot_velocity(self) -> Tuple[float, float, float]:
        """Get linear velocity (vx, vy, vz)."""
        vel, _ = p.getBaseVelocity(self._rover_id)
        return vel

    def set_motor_speeds(self, l1: float, l2: float, r1: float, r2: float):
        """Set mecanum wheel target velocities.

        Args:
            l1, l2, r1, r2: Wheel speeds in range -255..255 (matching hardware API).
        """
        # Normalize to m/s (approximate: 255 → ~0.5 m/s max)
        scale = 0.5 / 255.0
        self._motor_velocities = [l1 * scale, l2 * scale, r1 * scale, r2 * scale]

    def set_servo_pan(self, angle_deg: float):
        """Set camera pan angle (stored, used by SimCamera)."""
        self._pan_angle = max(config.SERVO_PAN_MIN,
                              min(config.SERVO_PAN_MAX, angle_deg))

    def set_servo_tilt(self, angle_deg: float):
        """Set camera tilt angle (stored, used by SimCamera)."""
        self._tilt_angle = max(config.SERVO_TILT_MIN,
                               min(config.SERVO_TILT_MAX, angle_deg))

    @property
    def pan_angle(self) -> float:
        return self._pan_angle

    @property
    def tilt_angle(self) -> float:
        return self._tilt_angle

    @property
    def rover_id(self) -> int:
        return self._rover_id

    @property
    def sim_time(self) -> float:
        return self._step_count * self._time_step

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(self._physics_client)

    # =========================================================================
    # World Construction
    # =========================================================================

    def _build_floor(self):
        """Create textured floor plane."""
        floor_id = p.loadURDF("plane.urdf")
        self._body_ids.append(floor_id)
        # Checkerboard floor color
        p.changeVisualShape(floor_id, -1, rgbaColor=(0.9, 0.85, 0.75, 1))

    def _build_walls(self):
        """Create walls from the floor plan definition."""
        for (x1, y1), (x2, y2) in self.plan["walls"]:
            wall_id = self._create_wall_segment(x1, y1, x2, y2)
            self._wall_ids.append(wall_id)
            self._body_ids.append(wall_id)

    def _create_wall_segment(self, x1, y1, x2, y2) -> int:
        """Create a single wall segment as a box."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cz = self.WALL_HEIGHT / 2.0

        half_extents = [length / 2.0, self.WALL_THICKNESS / 2.0, self.WALL_HEIGHT / 2.0]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                     rgbaColor=(0.85, 0.82, 0.78, 1))

        quat = p.getQuaternionFromEuler([0, 0, angle])
        body_id = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[cx, cy, cz],
            baseOrientation=quat,
        )
        return body_id

    def _build_furniture(self):
        """Add furniture items from the floor plan."""
        for item in self.plan.get("furniture", []):
            if item["type"] == "box":
                pos = item["pos"]
                size = item["size"]
                color = item.get("color", (0.5, 0.5, 0.5, 1))
                half = [s / 2.0 for s in size]
                col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                             rgbaColor=color)
                body_id = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[pos[0], pos[1], pos[2]])
                self._body_ids.append(body_id)

            elif item["type"] == "cylinder":
                pos = item["pos"]
                r = item["radius"]
                h = item["height"]
                color = item.get("color", (0.5, 0.5, 0.5, 1))
                col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
                vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                             rgbaColor=color)
                body_id = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[pos[0], pos[1], pos[2]])
                self._body_ids.append(body_id)

    def _build_rover(self, start_pose: Tuple[float, float, float]) -> int:
        """Create a simple box rover with mass."""
        x, y, theta = start_pose
        half = [self.ROVER_SIZE[0]/2, self.ROVER_SIZE[1]/2, self.ROVER_SIZE[2]/2]

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                     rgbaColor=(0.2, 0.2, 0.8, 1))

        quat = p.getQuaternionFromEuler([0, 0, theta])
        rover_id = p.createMultiBody(
            baseMass=self.ROVER_MASS,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[x, y, self.ROVER_HEIGHT + self.ROVER_SIZE[2]/2],
            baseOrientation=quat,
        )

        # Disable dynamics -- we control this kinematically
        # (wheeled robots on flat floors are better modeled kinematically)
        p.changeDynamics(rover_id, -1, mass=0)  # mass=0 makes it static/kinematic

        return rover_id

    def _apply_textures(self):
        """Apply visual textures to walls for feature-rich camera views.

        Creates procedural textures and applies them to wall surfaces so
        ORB can find features. Without textures, flat-color walls yield
        very few keypoints.
        """
        import tempfile
        rng = np.random.RandomState(42)

        for i, wall_id in enumerate(self._wall_ids):
            # Generate a unique procedural texture for each wall
            tex_img = self._generate_wall_texture(rng, wall_index=i)

            # Write to temp file and load as PyBullet texture
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tex_path = f.name
            import cv2
            cv2.imwrite(tex_path, tex_img)
            tex_id = p.loadTexture(tex_path)
            p.changeVisualShape(wall_id, -1, textureUniqueId=tex_id)
            os.unlink(tex_path)

        # Add texture to floor too
        floor_tex = self._generate_floor_texture(rng)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tex_path = f.name
        import cv2
        cv2.imwrite(tex_path, floor_tex)
        tex_id = p.loadTexture(tex_path)
        if self._body_ids:
            p.changeVisualShape(self._body_ids[0], -1, textureUniqueId=tex_id)
        os.unlink(tex_path)

    @staticmethod
    def _generate_wall_texture(rng, wall_index: int = 0,
                               size: int = 256) -> np.ndarray:
        """Generate a procedural wall texture with lots of visual features."""
        import cv2
        # Base color (slightly different per wall)
        base_colors = [(210, 200, 190), (195, 200, 210), (200, 195, 185), (205, 210, 200)]
        base = base_colors[wall_index % len(base_colors)]
        img = np.full((size, size, 3), base, dtype=np.uint8)

        # Add brick/panel pattern
        panel_h = rng.randint(30, 60)
        for y in range(0, size, panel_h):
            cv2.line(img, (0, y), (size, y), (170, 165, 160), 1)

        # Add random rectangles (picture frames, switches, etc.)
        for _ in range(rng.randint(3, 8)):
            x1 = rng.randint(10, size - 60)
            y1 = rng.randint(10, size - 60)
            w = rng.randint(20, 50)
            h = rng.randint(20, 50)
            color = tuple(int(c) for c in rng.randint(100, 230, 3))
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, -1)
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (80, 80, 80), 1)

        # Add circles (clocks, outlets)
        for _ in range(rng.randint(2, 5)):
            cx = rng.randint(20, size - 20)
            cy = rng.randint(20, size - 20)
            r = rng.randint(5, 15)
            color = tuple(int(c) for c in rng.randint(60, 200, 3))
            cv2.circle(img, (cx, cy), r, color, -1)

        # Add noise for fine texture
        noise = rng.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    @staticmethod
    def _generate_floor_texture(rng, size: int = 512) -> np.ndarray:
        """Generate a wood/tile floor texture."""
        import cv2
        img = np.full((size, size, 3), (180, 160, 130), dtype=np.uint8)

        # Tile pattern
        tile_size = 64
        for y in range(0, size, tile_size):
            for x in range(0, size, tile_size):
                shade = rng.randint(-20, 20)
                color = tuple(int(max(0, min(255, 180+shade+c))) for c in [-10, -5, 0])
                cv2.rectangle(img, (x, y), (x+tile_size-1, y+tile_size-1), color, -1)
                cv2.rectangle(img, (x, y), (x+tile_size-1, y+tile_size-1),
                              (140, 130, 110), 1)

        # Add grain noise
        noise = rng.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def _apply_motor_forces(self):
        """Move rover kinematically by directly updating position each step.

        With mass=0, the rover is a kinematic body. We compute desired
        velocity from mecanum kinematics and integrate position ourselves.
        Collision detection still works -- PyBullet prevents penetration.
        """
        l1, l2, r1, r2 = self._motor_velocities

        # Mecanum inverse kinematics (wheel speeds -> body velocity)
        vx = (l1 + l2 + r1 + r2) / 4.0            # forward (m/s)
        vy = (-l1 + l2 + r1 - r2) / 4.0            # lateral left (m/s)
        omega = (-l1 - l2 + r1 + r2) / (4.0 * 0.17)  # rotation (rad/s)

        # Get current pose
        pos, orn = p.getBasePositionAndOrientation(self._rover_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dt = self._time_step
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Integrate position
        dx_world = (vx * cos_y - vy * sin_y) * dt
        dy_world = (vx * sin_y + vy * cos_y) * dt
        dyaw = omega * dt

        new_x = pos[0] + dx_world
        new_y = pos[1] + dy_world
        new_yaw = yaw + dyaw
        new_z = self.ROVER_HEIGHT + self.ROVER_SIZE[2] / 2

        quat = p.getQuaternionFromEuler([0, 0, new_yaw])
        p.resetBasePositionAndOrientation(
            self._rover_id, [new_x, new_y, new_z], quat)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
