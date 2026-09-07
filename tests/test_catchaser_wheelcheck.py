"""The wheelcheck diagnostic must send the SAME wheel pattern the real
Actuators primitives do, or it would mislead the person watching the wheels."""

from catchaser.wheelcheck import PATTERNS
from raspbot_slam.actuators import Actuators


class _Bot:
    def __init__(self):
        self.m = {}

    def Ctrl_Muto(self, i, v):
        self.m[i] = v


def _wheels(fn):
    b = _Bot()
    fn(Actuators(bot=b))
    return (b.m[0], b.m[1], b.m[2], b.m[3])


def test_patterns_match_actuator_primitives():
    cases = {
        "forward": lambda a: a.move_forward(1),
        "back": lambda a: a.move_backward(1),
        "strafe_left": lambda a: a.move_left(1),
        "strafe_right": lambda a: a.move_right(1),
        "rotate_left": lambda a: a.rotate_left(1),
        "rotate_right": lambda a: a.rotate_right(1),
    }
    for name, fn in cases.items():
        assert _wheels(fn) == PATTERNS[name], name


def test_rotate_and_strafe_are_actually_different():
    # The user's confusion: they should NOT be the same wheel pattern.
    assert PATTERNS["rotate_right"] != PATTERNS["strafe_right"]
    # Rotate = left side one way, right side the other (spin).
    assert PATTERNS["rotate_right"] == (1, 1, -1, -1)
    # Strafe = diagonal pair pattern (translate).
    assert PATTERNS["strafe_right"] == (1, -1, -1, 1)
