"""
Run the SLAM pipeline in the PyBullet simulator.

This is the digital twin entry point: same SLAM code, simulated hardware.
Useful for testing, tuning parameters, and demos without the physical rover.

Usage:
    python -m raspbot_slam.simulator.run_sim --floor-plan L_shaped --gui
    python -m raspbot_slam.simulator.run_sim --floor-plan two_rooms --map-name sim_test
"""

import argparse
import os
import sys
import time
import numpy as np

from .. import config
from ..feature_extractor import FeatureExtractor
from ..visual_odometry import VisualOdometry
from ..synthetic_stereo import SyntheticStereo
from ..state_estimator import EKFSLAM
from ..map_manager import MapManager
from ..explorer import Explorer
from ..motion_controller import MotionController

from .sim_world import SimWorld, FLOOR_PLANS
from .sim_camera import SimCamera
from .sim_sensors import SimSensors
from .sim_actuators import SimActuators


def run_simulation(floor_plan: str = "L_shaped", gui: bool = False,
                   map_name: str = "sim_default", max_steps: int = 5000,
                   verbose: bool = True):
    """Run full SLAM mapping in simulation.

    Args:
        floor_plan: Name of floor plan ('simple_room', 'L_shaped', 'corridor', 'two_rooms').
        gui: Open PyBullet GUI window.
        map_name: Name for saved map.
        max_steps: Maximum simulation frames.
        verbose: Print progress.
    """
    if verbose:
        plan_info = FLOOR_PLANS.get(floor_plan, {})
        print(f"SLAM Simulation: {floor_plan}")
        print(f"  {plan_info.get('description', 'Unknown floor plan')}")
        print(f"  GUI: {gui}")
        print(f"  Max steps: {max_steps}")
        print()

    # Create simulated world
    world = SimWorld(floor_plan=floor_plan, gui=gui, start_pose=(0, 0, 0))

    # Create simulated hardware interfaces
    camera = SimCamera(world)
    sensors = SimSensors(world)
    actuators = SimActuators(world, realtime=gui)

    # Create SLAM pipeline (identical to real hardware usage)
    fe = FeatureExtractor()
    vo = VisualOdometry(camera, fe)
    stereo = SyntheticStereo(camera, fe, actuators, sensors, baseline_m=0.05)
    ekf = EKFSLAM(initial_pose=(0, 0, 0))
    map_mgr = MapManager()
    explorer = Explorer(map_mgr)
    mc = MotionController(actuators)

    # Mapping state
    distance_since_scan = 0.0
    keyframes_since_stereo = 0
    frame_count = 0
    t_start = time.time()

    if verbose:
        print("Starting mapping loop...")
        gt = world.get_robot_pose()
        print(f"  Ground truth start: ({gt[0]:.2f}, {gt[1]:.2f}, "
              f"{np.degrees(gt[2]):.1f}°)")
        print()

    try:
        while frame_count < max_steps:
            # Step physics
            world.step(4)  # 4 physics steps per frame

            # Capture and process
            frame = camera.capture()
            vo_delta = vo.process_frame(frame)

            if vo_delta is None:
                frame_count += 1
                continue

            dx, dy, dtheta = vo_delta
            ekf.predict(vo_delta)
            pose = ekf.get_pose()
            map_mgr.record_pose(*pose)

            displacement = np.sqrt(dx**2 + dy**2)
            distance_since_scan += displacement

            # Keyframe
            if vo.is_keyframe_needed():
                vo.create_keyframe()
                keyframes_since_stereo += 1

            # Ultrasonic
            us_range_mm = sensors.read_ultrasonic_mm()
            if us_range_mm > 0:
                map_mgr.occupancy.update_ultrasonic(
                    pose[0], pose[1], pose[2], us_range_mm / 1000.0)

            # Emergency stop
            if 0 < us_range_mm < config.OBSTACLE_STOP_MM:
                actuators.stop()
                world.step(20)

            # Scanning stop
            if distance_since_scan >= config.SCAN_INTERVAL_M:
                actuators.stop()
                world.step(20)

                # Synthetic stereo
                if keyframes_since_stereo >= config.STEREO_INTERVAL_KEYFRAMES:
                    observations = stereo.capture_and_triangulate()
                    keyframes_since_stereo = 0

                    for obs in observations:
                        if obs.confidence > 0.3:
                            cos_t = np.cos(pose[2])
                            sin_t = np.sin(pose[2])
                            wx = pose[0] + obs.position_3d[2] * cos_t - obs.position_3d[0] * sin_t
                            wy = pose[1] + obs.position_3d[2] * sin_t + obs.position_3d[0] * cos_t
                            wz = -obs.position_3d[1]
                            world_pos = np.array([wx, wy, wz])

                            existing = map_mgr.match_observation(
                                obs.descriptor, position_hint=world_pos,
                                max_spatial_distance=0.5)
                            if existing is not None:
                                map_mgr.update_landmark(existing.id, world_pos)
                            else:
                                lm_id = map_mgr.add_landmark(world_pos, obs.descriptor)
                                ekf.add_landmark(world_pos, obs.descriptor)

                    if observations:
                        pts = np.array([obs.position_3d for obs in observations])
                        map_mgr.occupancy.update_depth_points(
                            pose[0], pose[1], pose[2], pts)

                distance_since_scan = 0.0

            # Exploration target
            target = explorer.select_target(pose)
            if target is None:
                if verbose:
                    print(f"\n  No more frontiers at frame {frame_count}.")
                break

            mc.drive_to_waypoint(pose, target)
            frame_count += 1

            # Progress
            if verbose and frame_count % 100 == 0:
                gt = world.get_robot_pose()
                est = ekf.get_pose()
                err = np.sqrt((gt[0]-est[0])**2 + (gt[1]-est[1])**2)
                print(f"  Frame {frame_count:4d} | "
                      f"GT: ({gt[0]:5.2f}, {gt[1]:5.2f}) | "
                      f"EKF: ({est[0]:5.2f}, {est[1]:5.2f}) | "
                      f"Err: {err:.3f}m | "
                      f"LM: {map_mgr.landmark_count}")

    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user.")

    finally:
        actuators.stop()

    # Results
    elapsed = time.time() - t_start
    gt_final = world.get_robot_pose()
    est_final = ekf.get_pose()
    pos_error = np.sqrt((gt_final[0]-est_final[0])**2 + (gt_final[1]-est_final[1])**2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Simulation Complete")
        print(f"  Frames: {frame_count}")
        print(f"  Wall time: {elapsed:.1f}s")
        print(f"  Sim time: {world.sim_time:.1f}s")
        print(f"  Landmarks: {map_mgr.landmark_count}")
        print(f"  Trajectory poses: {len(map_mgr.trajectory)}")
        print(f"  Final position error: {pos_error:.3f}m")
        print(f"  Final heading error: "
              f"{abs(np.degrees(gt_final[2] - est_final[2])):.1f}°")

    # Save map
    map_dir = os.path.join(config.DEFAULT_MAP_DIR, map_name)
    map_mgr.save(map_dir)
    if verbose:
        print(f"\n  Map saved to {map_dir}")
        print(f"  Visualize: python -m raspbot_slam.visualize_map {map_dir}")

    world.close()

    return {
        "frames": frame_count,
        "landmarks": map_mgr.landmark_count,
        "position_error_m": pos_error,
        "heading_error_deg": abs(np.degrees(gt_final[2] - est_final[2])),
        "map_dir": map_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="RASPBOT-V2 SLAM Simulator")
    parser.add_argument("--floor-plan", default="L_shaped",
                        choices=list(FLOOR_PLANS.keys()),
                        help="Floor plan to simulate")
    parser.add_argument("--gui", action="store_true",
                        help="Open PyBullet GUI window")
    parser.add_argument("--map-name", default="sim_default",
                        help="Name for saved map")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum simulation frames")
    args = parser.parse_args()

    run_simulation(
        floor_plan=args.floor_plan,
        gui=args.gui,
        map_name=args.map_name,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
