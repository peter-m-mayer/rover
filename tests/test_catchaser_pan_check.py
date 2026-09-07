"""Tests for the pan-servo sign check (pure logic; hardware parts not tested)."""

from catchaser.pan_check import recommend_sign


def test_scene_shifts_right_means_camera_left_sign_plus():
    # Increasing pan shifts the static scene right -> camera turned left ->
    # simulator convention -> CHASE_PAN_SIGN = +1.
    assert recommend_sign(+12.4) == 1


def test_scene_shifts_left_means_camera_right_sign_minus():
    assert recommend_sign(-9.1) == -1
