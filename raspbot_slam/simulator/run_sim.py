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

    def rotation_scan(world, actuators, sensors, map_mgr, n_steps=12):
        """Rotate in place, firing ultrasonic at each step to map surroundings."""
        step_angle = 360.0 / n_steps
        for _ in range(n_steps):
            actuators.rotate_right(50)
            world.step(int(240 * step_angle / 360 / 0.7))  # rough timing for one step
            actuators.stop()
            world.step(10)
            gt = world.get_robot_pose()
            us = sensors.read_ultrasonic_mm()
            if us > 0:
                map_mgr.occupancy.update_ultrasonic(
                    gt[0], gt[1], gt[2], us / 1000.0)
                map_mgr.occupancy.mark_traversed(gt[0], gt[1])

    if verbose:
        print("Starting mapping loop...")
        gt = world.get_robot_pose()
        print(f"  Ground truth start: ({gt[0]:.2f}, {gt[1]:.2f}, "
              f"{np.degrees(gt[2]):.1f}°)")
        print()

    # Initial 360-degree rotation scan to bootstrap the occupancy grid
    if verbose:
        print("  Initial rotation scan...")
    rotation_scan(world, actuators, sensors, map_mgr)

    # Track ground-truth distance for scan triggers (not VO distance which
    # is unreliable before scale calibration)
    gt_prev = world.get_robot_pose()
    gt_distance_since_scan = 0.0
    gt_distance_since_rotation = 0.0
    stereo_count = 0

    # VO scale calibration: track cumulative VO displacement between stereo stops
    vo_displacement_since_stereo = 0.0
    gt_displacement_since_stereo = 0.0
    scale_calibrated = False

    try:
        while frame_count < max_steps:
            # Step physics (8 steps = 1/30s at 240Hz — roughly camera framerate)
            world.step(8)

            # Capture and process frame through VO
            frame = camera.capture()
            vo_delta = vo.process_frame(frame)

            if vo_delta is None:
                frame_count += 1
                continue

            dx, dy, dtheta = vo_delta

            # Track VO displacement in RAW units (before scale) for calibration.
            # dx,dy are already scaled by vo.scale, so divide it back out.
            if vo.scale > 1e-8:
                raw_vo_step = np.sqrt(dx**2 + dy**2) / vo.scale
            else:
                raw_vo_step = 0
            vo_displacement_since_stereo += raw_vo_step

            # EKF prediction with VO delta
            ekf.predict(vo_delta)
            pose = ekf.get_pose()
            map_mgr.record_pose(*pose)

            # Track ground-truth distance for scan interval
            gt_now = world.get_robot_pose()
            gt_step = np.sqrt((gt_now[0]-gt_prev[0])**2 + (gt_now[1]-gt_prev[1])**2)
            gt_distance_since_scan += gt_step
            gt_distance_since_rotation += gt_step
            gt_displacement_since_stereo += gt_step
            gt_prev = gt_now

            # Periodic rotation scan (every 1m) to discover lateral space
            if gt_distance_since_rotation >= 1.0:
                rotation_scan(world, actuators, sensors, map_mgr, n_steps=8)
                gt_distance_since_rotation = 0.0
                # Reset VO after rotation (scene has changed dramatically)
                vo.reset()
                vo.process_frame(camera.capture())

            # Keyframe
            if vo.is_keyframe_needed():
                vo.create_keyframe()
                keyframes_since_stereo += 1

            # Ultrasonic → occupancy grid (every frame, cheap)
            us_range_mm = sensors.read_ultrasonic_mm()
            if us_range_mm > 0:
                # Use ground truth pose for occupancy (EKF pose is unreliable early on)
                map_mgr.occupancy.update_ultrasonic(
                    gt_now[0], gt_now[1], gt_now[2], us_range_mm / 1000.0)

            # Emergency stop
            if 0 < us_range_mm < config.OBSTACLE_STOP_MM:
                actuators.stop()
                world.step(20)
                frame_count += 1
                continue

            # Synthetic stereo stop (triggered by real distance, not VO distance)
            if gt_distance_since_scan >= config.SCAN_INTERVAL_M:
                actuators.stop()
                world.step(20)

                observations = stereo.capture_and_triangulate()
                stereo_count += 1

                # Add landmarks from depth (limit to best 20 per stop to avoid flooding)
                obs_sorted = sorted(
                    [o for o in observations if o.confidence > 0.3],
                    key=lambda o: -o.confidence)[:20]

                for obs in obs_sorted:
                    cos_t = np.cos(gt_now[2])
                    sin_t = np.sin(gt_now[2])
                    wx = gt_now[0] + obs.position_3d[2] * cos_t - obs.position_3d[0] * sin_t
                    wy = gt_now[1] + obs.position_3d[2] * sin_t + obs.position_3d[0] * cos_t
                    wz = -obs.position_3d[1]
                    world_pos = np.array([wx, wy, wz])

                    existing = map_mgr.match_observation(
                        obs.descriptor, position_hint=world_pos,
                        max_spatial_distance=0.5)
                    if existing is not None:
                        map_mgr.update_landmark(existing.id, world_pos)
                    else:
                        map_mgr.add_landmark(world_pos, obs.descriptor)

                # Update occupancy from depth
                if observations:
                    pts = np.array([obs.position_3d for obs in observations])
                    map_mgr.occupancy.update_depth_points(
                        gt_now[0], gt_now[1], gt_now[2], pts)

                # VO scale calibration: ratio of real distance to VO distance
                # between consecutive stereo stops. This is the key step that
                # gives the VO absolute scale.
                if vo_displacement_since_stereo > 1e-6 and gt_displacement_since_stereo > 0.05:
                    new_scale = (gt_displacement_since_stereo / vo_displacement_since_stereo)
                    if scale_calibrated:
                        # Smooth update (80% old, 20% new)
                        vo.scale = 0.8 * vo.scale + 0.2 * new_scale
                    else:
                        vo.scale = new_scale
                        scale_calibrated = True

                    # Note: EKF scale stays at 1.0 because VO deltas are
                    # already scaled. The EKF scale factor is for additional
                    # correction only.

                    if verbose and stereo_count <= 5:
                        print(f"    Scale calibration: GT={gt_displacement_since_stereo:.3f}m / "
                              f"VO_raw={vo_displacement_since_stereo:.3f} → "
                              f"new_scale={new_scale:.6f}, "
                              f"applied_scale={vo.scale:.6f}")

                vo_displacement_since_stereo = 0.0
                gt_displacement_since_stereo = 0.0
                gt_distance_since_scan = 0.0

            # Exploration: plan path to best frontier, follow waypoints.
            if not hasattr(run_simulation, '_path'):
                run_simulation._path = []
                run_simulation._replan_counter = 0

            run_simulation._replan_counter += 1

            # Re-plan every 50 frames or when path is exhausted
            need_replan = (not run_simulation._path or
                           run_simulation._replan_counter >= 50)

            if need_replan:
                target = explorer.select_target(gt_now[:2] + (gt_now[2],))
                if target is None:
                    if verbose:
                        print(f"\n  No more frontiers at frame {frame_count}.")
                    break
                new_path = explorer.plan_path(gt_now[:2], target)
                # Skip waypoints already within reach
                while new_path:
                    d = np.sqrt((new_path[0][0]-gt_now[0])**2 +
                                (new_path[0][1]-gt_now[1])**2)
                    if d < config.WAYPOINT_TOLERANCE_M * 2:
                        new_path.pop(0)
                    else:
                        break
                if new_path:
                    run_simulation._path = new_path
                run_simulation._replan_counter = 0

            # Follow the current path
            if run_simulation._path:
                wp = run_simulation._path[0]
                reached = mc.drive_to_waypoint(gt_now, wp)
                if reached:
                    run_simulation._path.pop(0)
            else:
                # No viable path — drive forward to explore
                actuators.move_forward(config.NAV_SPEED)

            frame_count += 1

            # Progress
            if verbose and frame_count % 50 == 0:
                gt = world.get_robot_pose()
                est = ekf.get_pose()
                err = np.sqrt((gt[0]-est[0])**2 + (gt[1]-est[1])**2)
                print(f"  Frame {frame_count:4d} | "
                      f"GT: ({gt[0]:5.2f}, {gt[1]:5.2f}) | "
                      f"EKF: ({est[0]:5.2f}, {est[1]:5.2f}) | "
                      f"Err: {err:.3f}m | "
                      f"LM: {map_mgr.landmark_count:4d} | "
                      f"Stereo: {stereo_count}")

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
