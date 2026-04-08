"""
Main entry point: autonomous mapping mode.

The rover explores the environment using frontier-based exploration,
building a sparse landmark map and occupancy grid.

Usage (on the Pi):
    python -m raspbot_slam.run_mapping --map-name ground_floor

The robot will:
1. Calibrate (if needed)
2. Explore autonomously, stopping periodically for stereo depth
3. Save the map when exploration is complete or interrupted (Ctrl+C)
"""

import argparse
import os
import signal
import sys
import time
import numpy as np

from . import config
from .camera import Camera
from .feature_extractor import FeatureExtractor
from .visual_odometry import VisualOdometry
from .synthetic_stereo import SyntheticStereo
from .state_estimator import EKFSLAM
from .map_manager import MapManager
from .explorer import Explorer
from .motion_controller import MotionController
from .actuators import Actuators
from .sensors import Sensors


class MappingSession:
    """Orchestrates the full mapping pipeline."""

    def __init__(self, map_name: str = "default", bot=None):
        self.map_name = map_name
        self.map_dir = os.path.join(config.DEFAULT_MAP_DIR, map_name)

        # Hardware
        self.camera = Camera()
        self.actuators = Actuators(bot)
        self.sensors = Sensors(bot)

        # SLAM pipeline
        self.fe = FeatureExtractor()
        self.vo = VisualOdometry(self.camera, self.fe)
        self.stereo = SyntheticStereo(self.camera, self.fe, self.actuators, self.sensors)
        self.ekf = EKFSLAM(initial_pose=(0, 0, 0))
        self.map_mgr = MapManager()
        self.explorer = Explorer(self.map_mgr)
        self.mc = MotionController(self.actuators)

        # State tracking
        self.distance_since_scan = 0.0
        self.keyframes_since_stereo = 0
        self.running = False
        self.frame_count = 0
        self.start_time = None

    def run(self):
        """Main mapping loop."""
        self.running = True
        self.start_time = time.time()

        # Setup
        self.camera.open()
        self.sensors.enable_ultrasonic()
        self.actuators.center_camera()
        self.actuators.set_led_color("mapping")

        print(f"Mapping session '{self.map_name}' started.")
        print("Press Ctrl+C to stop and save.\n")

        # Initial scan
        print("Performing initial scan...")
        self._scanning_stop()

        try:
            while self.running:
                self._mapping_step()
                self.frame_count += 1

                # Check if mapping is complete
                if self.explorer.is_mapping_complete():
                    print("\nMapping complete! No more frontiers to explore.")
                    break

        except KeyboardInterrupt:
            print("\n\nMapping interrupted by user.")

        finally:
            self._shutdown()

    def _mapping_step(self):
        """One iteration of the mapping loop."""
        # Capture and process frame
        frame = self.camera.capture()
        vo_delta = self.vo.process_frame(frame)

        if vo_delta is None:
            # Tracking lost
            self.actuators.set_led_color("lost")
            self.actuators.stop()
            time.sleep(0.5)
            self.actuators.set_led_color("mapping")
            return

        dx, dy, dtheta = vo_delta

        # EKF prediction
        self.ekf.predict(vo_delta)
        pose = self.ekf.get_pose()

        # Record trajectory
        self.map_mgr.record_pose(*pose)

        # Track distance for scan intervals
        displacement = np.sqrt(dx**2 + dy**2)
        self.distance_since_scan += displacement

        # Keyframe management
        if self.vo.is_keyframe_needed():
            kf = self.vo.create_keyframe()
            if kf is not None:
                self.keyframes_since_stereo += 1
                self._save_keyframe(kf)

        # Ultrasonic obstacle check
        us_range = self.sensors.read_ultrasonic_mm()
        if 0 < us_range < config.OBSTACLE_STOP_MM:
            self.actuators.stop()
            self.map_mgr.occupancy.update_ultrasonic(
                pose[0], pose[1], pose[2], us_range / 1000.0)
        elif us_range > 0:
            self.map_mgr.occupancy.update_ultrasonic(
                pose[0], pose[1], pose[2], us_range / 1000.0)

        # Scanning stop trigger
        if self.distance_since_scan >= config.SCAN_INTERVAL_M:
            self._scanning_stop()

        # Select exploration target and drive
        target = self.explorer.select_target(pose)
        if target is not None:
            self.mc.drive_to_waypoint(pose, target)

        # Periodic status
        if self.frame_count % 50 == 0:
            elapsed = time.time() - self.start_time
            print(f"  Frame {self.frame_count} | "
                  f"Pose: ({pose[0]:.2f}, {pose[1]:.2f}, {np.degrees(pose[2]):.1f}°) | "
                  f"Landmarks: {self.map_mgr.landmark_count} | "
                  f"Time: {elapsed:.0f}s")

    def _scanning_stop(self):
        """Perform a full scanning stop: pan sweep + synthetic stereo + ultrasonic."""
        self.actuators.stop()
        self.actuators.set_led_color("scanning")
        time.sleep(0.1)

        pose = self.ekf.get_pose()

        # Pan sweep: capture features at multiple angles
        for pan_angle in config.PAN_SWEEP_ANGLES:
            self.actuators.set_servo_pan(pan_angle)
            time.sleep(config.SERVO_SETTLE_MS / 1000.0)

            frame = self.camera.capture()
            kp, desc = self.fe.detect_and_compute(frame)

            if desc is not None:
                visible = self.map_mgr.get_visible_landmarks(pose, fov_deg=60)
                for i, d in enumerate(desc):
                    match = self.map_mgr.match_observation(d)
                    if match is not None:
                        # Update existing landmark
                        dx = match.position_3d[0] - pose[0]
                        dy = match.position_3d[1] - pose[1]
                        bearing = np.arctan2(dy, dx) - pose[2]
                        range_m = np.sqrt(dx**2 + dy**2)
                        self.ekf.update_landmark(match.id, bearing, range_m, d)

        # Return camera to center for stereo
        self.actuators.center_camera()
        time.sleep(0.2)

        # Synthetic stereo
        if self.keyframes_since_stereo >= config.STEREO_INTERVAL_KEYFRAMES:
            observations = self.stereo.capture_and_triangulate()
            self.keyframes_since_stereo = 0

            # Add new landmarks from depth observations
            for obs in observations:
                if obs.confidence > 0.3:
                    # Transform from camera frame to world frame
                    cos_t = np.cos(pose[2])
                    sin_t = np.sin(pose[2])
                    wx = pose[0] + obs.position_3d[2] * cos_t - obs.position_3d[0] * sin_t
                    wy = pose[1] + obs.position_3d[2] * sin_t + obs.position_3d[0] * cos_t
                    wz = -obs.position_3d[1]  # camera Y is down

                    world_pos = np.array([wx, wy, wz])

                    # Check if this matches an existing landmark
                    existing = self.map_mgr.match_observation(
                        obs.descriptor, position_hint=world_pos,
                        max_spatial_distance=0.5)

                    if existing is not None:
                        self.map_mgr.update_landmark(existing.id, world_pos)
                    else:
                        lm_id = self.map_mgr.add_landmark(world_pos, obs.descriptor)
                        self.ekf.add_landmark(world_pos, obs.descriptor)

            # Update occupancy grid with depth points
            if observations:
                pts = np.array([obs.position_3d for obs in observations])
                self.map_mgr.occupancy.update_depth_points(
                    pose[0], pose[1], pose[2], pts)

            # Cross-validate scale with ultrasonic
            us_range = self.sensors.read_ultrasonic_mm()
            if us_range > 0 and observations:
                new_scale = self.stereo.cross_validate_with_ultrasonic(
                    observations, us_range)

        self.distance_since_scan = 0.0
        self.actuators.set_led_color("mapping")

    def _save_keyframe(self, kf):
        """Save a keyframe to disk for offline processing."""
        kf_dir = os.path.join(self.map_dir, "keyframes")
        os.makedirs(kf_dir, exist_ok=True)

        try:
            import cv2
            prefix = os.path.join(kf_dir, f"kf_{kf.id:04d}")
            cv2.imwrite(f"{prefix}.jpg", kf.frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            np.save(f"{prefix}_desc.npy", kf.descriptors)
            np.save(f"{prefix}_kp.npy", kf.keypoints_xy)
        except Exception:
            pass  # non-critical

    def _shutdown(self):
        """Clean up and save."""
        self.actuators.stop()
        self.actuators.set_led_color("off")
        self.actuators.center_camera()
        self.camera.close()
        self.sensors.disable_ultrasonic()

        # Save map
        print(f"\nSaving map to {self.map_dir}...")
        self.map_mgr.save(self.map_dir)

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Done! {self.frame_count} frames, "
              f"{self.map_mgr.landmark_count} landmarks, "
              f"{elapsed:.0f} seconds.")
        print(f"\nVisualize with:")
        print(f"  python -m raspbot_slam.visualize_map {self.map_dir}")


def main():
    parser = argparse.ArgumentParser(description="RASPBOT-V2 Autonomous Mapping")
    parser.add_argument("--map-name", default="default",
                        help="Name for the map (creates maps/<name>/ directory)")
    args = parser.parse_args()

    session = MappingSession(map_name=args.map_name)
    session.run()


if __name__ == "__main__":
    main()
