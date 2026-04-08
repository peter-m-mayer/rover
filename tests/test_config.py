"""Tests for config.py -- parameter validation."""

from raspbot_slam import config


def test_orb_params_valid():
    assert config.ORB_N_FEATURES > 0
    assert 1.0 < config.ORB_SCALE_FACTOR < 2.0
    assert config.ORB_N_LEVELS >= 1


def test_lowe_ratio_in_range():
    assert 0.0 < config.LOWE_RATIO_THRESHOLD < 1.0
    assert 0.0 < config.LOWE_RATIO_LOOP_CLOSURE < 1.0
    assert config.LOWE_RATIO_LOOP_CLOSURE <= config.LOWE_RATIO_THRESHOLD


def test_min_inliers_positive():
    assert config.MIN_INLIER_MATCHES >= 5


def test_keyframe_thresholds_positive():
    assert config.KEYFRAME_TRANSLATION_M > 0
    assert config.KEYFRAME_ROTATION_DEG > 0
    assert 0 < config.KEYFRAME_OVERLAP_THRESHOLD < 1


def test_ekf_params():
    assert config.MAX_ACTIVE_LANDMARKS > 0
    assert 0 < config.PROCESS_NOISE_POSITION < 1
    assert 0 < config.PROCESS_NOISE_HEADING < 1
    assert config.CHI_SQUARED_GATE > 0


def test_grid_params():
    assert config.GRID_RESOLUTION_M > 0
    assert config.GRID_SIZE_CELLS > 0
    assert config.LOG_ODDS_FREE < 0
    assert config.LOG_ODDS_OCCUPIED > 0
    assert config.LOG_ODDS_MIN < config.LOG_ODDS_MAX


def test_nav_params():
    assert config.WAYPOINT_TOLERANCE_M > 0
    assert config.OBSTACLE_STOP_MM > 0
    assert 0 < config.NAV_SPEED <= config.MOTOR_SPEED_MAX


def test_servo_limits():
    assert config.SERVO_PAN_MIN <= config.SERVO_PAN_CENTER <= config.SERVO_PAN_MAX
    assert config.SERVO_TILT_MIN <= config.SERVO_TILT_REST <= config.SERVO_TILT_MAX


def test_stereo_params():
    assert config.STRAFE_DISTANCE_MM > 0
    assert config.STRAFE_SPEED > 0
    assert config.MIN_DISPARITY_PX >= 1
    assert config.STEREO_INTERVAL_KEYFRAMES >= 1
