"""Validate the chase controller in the PyBullet digital twin — before hardware.

The real cat detector (YOLOv8n) can't recognize a PyBullet primitive as a
"cat", so we can't point ONNX at rendered frames. Instead we validate the
part that is actually new and risky — the *control loop* — with a synthetic
detector that projects a virtual cat's true bearing into a pixel centroid,
exactly matching SimCamera's field of view. The cat is also a real physical
box in the world, so the simulated ultrasonic ray-cast detects it naturally
and the 200 mm emergency stop is exercised end-to-end.

This drives the same ChaseController.run() loop the robot will use.

    python -m catchaser.chase_sim                     # headless, approach scenario
    python -m catchaser.chase_sim --scenario search   # cat starts out of frame
    python -m catchaser.chase_sim --gui               # watch it
"""

import argparse
import math
import sys

import pybullet as p

from raspbot_slam import config
from raspbot_slam.simulator import SimWorld, SimCamera, SimSensors, SimActuators

from .chase import (ChaseController, STATE_HOLD, STATE_SEARCHING, STATE_TRACKING,
                    STATE_DART, STATE_FREEZE, STATE_FLEE)
from .detector import COCO_CAT, Detection


CAT_HALF = (0.10, 0.10, 0.13)   # box half-extents (m); ~0.26 m tall, seen by ultrasonic
CAT_COLOR = (0.85, 0.45, 0.15, 1.0)   # ginger


def add_cat(pos_xy):
    """Spawn a physical 'cat' box in the world; return (body_id, (x, y))."""
    x, y = pos_xy
    z = CAT_HALF[2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=CAT_HALF)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=CAT_HALF, rgbaColor=CAT_COLOR)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=[x, y, z])
    return body, (x, y)


def make_synthetic_detector(world: SimWorld, camera: SimCamera, cat_xy,
                            max_range_m: float = 6.0):
    """Build a perceive() that returns a Detection for the cat iff it's in view.

    Projects the cat's bearing (relative to the camera's optical axis, incl.
    pan) to a pixel centroid using the camera's own focal length — so the
    controller sees the same geometry SimCamera would render.
    """
    W, H = camera.width, camera.height
    fx = camera.K[0, 0]                       # px; square pixels in sim
    half_hfov = math.atan((W / 2.0) / fx)     # horizontal half-FOV
    cat_x, cat_y = cat_xy
    cat_h = 2 * CAT_HALF[2]

    def perceive():
        rx, ry, yaw = world.get_robot_pose()
        cam_yaw = yaw + math.radians(world.pan_angle - config.SERVO_PAN_CENTER)
        dx, dy = cat_x - rx, cat_y - ry
        dist = math.hypot(dx, dy)
        # Body-frame components relative to the camera axis.
        fwd = dx * math.cos(cam_yaw) + dy * math.sin(cam_yaw)
        left = -dx * math.sin(cam_yaw) + dy * math.cos(cam_yaw)
        if fwd <= 0.05 or dist > max_range_m:
            return []                          # behind camera or too far
        bearing = math.atan2(left, fwd)        # + = cat to the left
        if abs(bearing) > half_hfov:
            return []                          # outside horizontal FOV
        u = W / 2.0 - fx * math.tan(bearing)   # left bearing -> smaller u
        v = H / 2.0
        box_h = max(8.0, fx * cat_h / max(dist, 0.1))
        box_w = box_h * (CAT_HALF[0] / CAT_HALF[2])
        return [Detection(u - box_w / 2, v - box_h / 2,
                          u + box_w / 2, v + box_h / 2,
                          confidence=0.9, class_id=COCO_CAT)]

    return perceive


SCENARIOS = {
    # cat visible ahead-ish; rover must center and approach to 200 mm.
    "approach": {"rover": (0.0, 0.0, 0.0), "cat": (1.5, 0.4)},
    # cat starts outside the FOV to the right; rover must search-spin to find it.
    "search":   {"rover": (0.0, 0.0, 0.0), "cat": (0.4, -1.2)},
}


def run_chase_sim(scenario: str = "approach", gui: bool = False,
                  max_frames: int = 400, period_s: float = 0.1,
                  floor_plan: str = "simple_room", verbose: bool = False,
                  use_pan: bool = False, mode: str = "chase") -> dict:
    """Run one closed-loop chase in the simulator and return a result summary."""
    spec = SCENARIOS[scenario]
    world = SimWorld(floor_plan=floor_plan, gui=gui, start_pose=spec["rover"])
    steps_per_iter = max(1, int(period_s / world._time_step))
    try:
        _, cat_xy = add_cat(spec["cat"])
        camera = SimCamera(world)
        sensors = SimSensors(world)
        actuators = SimActuators(world)
        controller = ChaseController(actuators, sensors, use_pan=use_pan, mode=mode)
        perceive = make_synthetic_detector(world, camera, cat_xy)

        def range_to_cat_mm():
            rx, ry, _ = world.get_robot_pose()
            surf = math.hypot(cat_xy[0] - rx, cat_xy[1] - ry) - CAT_HALF[0]
            return int(max(0.0, surf) * 1000)

        trajectory = []
        states = []
        errors = []
        min_range = [float("inf")]

        def on_step(cmd):
            world.step(steps_per_iter)
            rx, ry, yaw = world.get_robot_pose()
            trajectory.append((rx, ry, yaw))
            states.append(cmd.state)
            if cmd.state in (STATE_TRACKING, STATE_HOLD, STATE_DART,
                             STATE_FREEZE, STATE_FLEE):
                errors.append(abs(cmd.heading_error))
            min_range[0] = min(min_range[0], range_to_cat_mm())
            if verbose:
                pan = f" pan={cmd.pan:5.1f}" if cmd.pan is not None else ""
                strafe = f" str={cmd.strafe:+5.1f}" if cmd.strafe else ""
                print(f"  {cmd.state:9s} err={cmd.heading_error:+.2f} "
                      f"fwd={cmd.forward:5.1f} turn={cmd.turn:+6.1f}{strafe}{pan} "
                      f"range={range_to_cat_mm():4d}mm")

        # Use sim time as the clock; don't wall-sleep unless showing the GUI.
        controller.run(
            perceive, sensors.read_ultrasonic_mm,
            on_step=on_step, max_frames=max_frames, period_s=period_s,
            now=lambda: world.sim_time,
            sleep=(__import__("time").sleep if gui else (lambda s: None)),
        )

        final_range = range_to_cat_mm()
        rx, ry, _ = world.get_robot_pose()
        return {
            "scenario": scenario,
            "final_range_mm": final_range,
            "min_range_mm": int(min_range[0]),
            "stop_mm": controller.stop_mm,
            # Success: closed to the stop band and never rammed the cat.
            "reached": final_range <= controller.stop_mm + 60,
            "no_collision": min_range[0] >= 30,   # never drove into the box
            "acquired": STATE_TRACKING in states or STATE_HOLD in states
                        or STATE_DART in states or STATE_FREEZE in states
                        or STATE_FLEE in states,
            "searched": STATE_SEARCHING in states,
            "darted": STATE_DART in states,
            "froze": STATE_FREEZE in states,
            "fled": STATE_FLEE in states,
            "mean_abs_err": round(sum(errors) / len(errors), 3) if errors else None,
            "frames": len(states),
            "final_xy": (round(rx, 3), round(ry, 3)),
            "trajectory": trajectory,
            "states": states,
        }
    finally:
        world.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chase controller sim validation")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="approach")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--floor-plan", default="simple_room")
    ap.add_argument("--pan", action="store_true", help="camera pan tracking")
    ap.add_argument("--prey", action="store_true", help="prey/play mode")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    mode = "prey" if args.prey else "chase"
    r = run_chase_sim(args.scenario, gui=args.gui, max_frames=args.max_frames,
                      floor_plan=args.floor_plan, verbose=args.verbose,
                      use_pan=args.pan, mode=mode)
    print(f"\nscenario={r['scenario']}  mode={mode}  pan={args.pan}  frames={r['frames']}")
    print(f"  acquired cat : {r['acquired']}   searched: {r['searched']}")
    print(f"  mean |err|   : {r['mean_abs_err']}  (tracking tightness)")
    print(f"  min range    : {r['min_range_mm']} mm  (no-collision: {r['no_collision']})")
    print(f"  final xy     : {r['final_xy']}")
    if mode == "prey":
        print(f"  prey moves   : darted={r['darted']} froze={r['froze']} fled={r['fled']}")
        ok = r["acquired"] and r["no_collision"] and r["froze"]
    else:
        print(f"  final range  : {r['final_range_mm']} mm  (stop at {r['stop_mm']} mm)")
        ok = r["reached"] and r["no_collision"] and r["acquired"]
    print("RESULT:", "PASS." if ok else "FAIL — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
