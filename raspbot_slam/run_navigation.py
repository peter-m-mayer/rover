"""
Main entry point: navigation mode on a stored map.

Loads a previously built map, relocalizes the robot, and navigates
to commanded goals.

Usage (on the Pi):
    python -m raspbot_slam.run_navigation --map-name ground_floor --goal 3.0 2.5
"""

import argparse
import os
import sys
import time
import numpy as np

from . import config
from .camera import Camera
from .feature_extractor import FeatureExtractor
from .visual_odometry import VisualOdometry
from .state_estimator import EKFSLAM
from .map_manager import MapManager
from .motion_controller import MotionController
from .navigator import Navigator
from .actuators import Actuators
from .sensors import Sensors


class NavigationSession:
    """Orchestrates localization and goal-directed navigation."""

    def __init__(self, map_name: str, bot=None):
        self.map_dir = os.path.join(config.DEFAULT_MAP_DIR, map_name)
        if not os.path.isdir(self.map_dir):
            print(f"ERROR: Map directory not found: {self.map_dir}")
            print("Run mapping first: python -m raspbot_slam.run_mapping --map-name " + map_name)
            sys.exit(1)

        # Hardware
        self.camera = Camera()
        self.actuators = Actuators(bot)
        self.sensors = Sensors(bot)

        # SLAM pipeline
        self.fe = FeatureExtractor()
        self.vo = VisualOdometry(self.camera, self.fe)
        self.ekf = EKFSLAM()
        self.map_mgr = MapManager()
        self.mc = MotionController(self.actuators)
        self.navigator = Navigator(
            self.camera, self.fe, self.actuators, self.sensors,
            self.map_mgr, self.ekf, self.mc)

    def run(self, goal_x: float, goal_y: float):
        """Load map, relocalize, navigate to goal."""
        # Load map
        print(f"Loading map from {self.map_dir}...")
        self.map_mgr.load(self.map_dir)
        print(f"  Loaded {self.map_mgr.landmark_count} landmarks, "
              f"grid {self.map_mgr.occupancy.size}x{self.map_mgr.occupancy.size}")

        # Setup hardware
        self.camera.open()
        self.sensors.enable_ultrasonic()
        self.actuators.center_camera()
        self.actuators.set_led_color("lost")

        try:
            # Relocalize
            print("\nRelocalizing...")
            if not self.navigator.relocalize():
                print("ERROR: Could not determine robot position on map.")
                print("Try repositioning the robot near a feature-rich area.")
                return

            pose = self.ekf.get_pose()
            print(f"  Localized at ({pose[0]:.2f}, {pose[1]:.2f}, "
                  f"{np.degrees(pose[2]):.1f}°)")
            self.actuators.set_led_color("localizing")

            # Navigate to goal
            print(f"\nNavigating to ({goal_x:.2f}, {goal_y:.2f})...")
            self.actuators.beep(0.2)

            # Main navigation loop with continuous VO updates
            goal = (goal_x, goal_y)
            reached = False
            frame_count = 0

            while not reached:
                # Capture and process frame for VO
                frame = self.camera.capture()
                vo_delta = self.vo.process_frame(frame)

                # Update localization
                self.navigator.update_tracking(frame, vo_delta)

                if not self.navigator.is_localized:
                    print("  Tracking lost! Attempting relocalization...")
                    self.actuators.stop()
                    if not self.navigator.relocalize():
                        print("  Relocalization failed. Stopping.")
                        break
                    print("  Relocalized!")

                # Navigate
                pose = self.ekf.get_pose()
                dx = goal_x - pose[0]
                dy = goal_y - pose[1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist < config.WAYPOINT_TOLERANCE_M:
                    reached = True
                    break

                # Obstacle check
                if self.sensors.is_obstacle_ahead():
                    self.mc.stop()
                    us = self.sensors.read_ultrasonic_m()
                    self.map_mgr.occupancy.update_ultrasonic(
                        pose[0], pose[1], pose[2], us)

                self.mc.drive_to_waypoint(pose, goal)

                # Status
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"  Pose: ({pose[0]:.2f}, {pose[1]:.2f}) | "
                          f"Distance to goal: {dist:.2f}m")

            if reached:
                self.actuators.stop()
                self.actuators.beep(0.1)
                time.sleep(0.1)
                self.actuators.beep(0.1)
                pose = self.ekf.get_pose()
                print(f"\nGoal reached! Final pose: "
                      f"({pose[0]:.2f}, {pose[1]:.2f}, {np.degrees(pose[2]):.1f}°)")
            else:
                print("\nNavigation failed or was interrupted.")

        except KeyboardInterrupt:
            print("\n\nNavigation interrupted by user.")

        finally:
            self.actuators.stop()
            self.actuators.set_led_color("off")
            self.actuators.center_camera()
            self.camera.close()
            self.sensors.disable_ultrasonic()


def main():
    parser = argparse.ArgumentParser(description="RASPBOT-V2 Map Navigation")
    parser.add_argument("--map-name", required=True,
                        help="Name of the map to load (from maps/<name>/)")
    parser.add_argument("--goal", nargs=2, type=float, required=True,
                        metavar=("X", "Y"),
                        help="Goal position in meters (world frame)")
    args = parser.parse_args()

    session = NavigationSession(map_name=args.map_name)
    session.run(args.goal[0], args.goal[1])


if __name__ == "__main__":
    main()
