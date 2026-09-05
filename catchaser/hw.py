"""Hardware connection helper for Cat Chaser 3000.

Centralizes the one piece of platform-specific glue: importing the Yahboom
vendor driver (installed on the Pi via its setup.py, not tracked in git) and
constructing the shared hardware interfaces from raspbot_slam.

Everything degrades gracefully to mock mode (bot=None) so the full stack
runs on any development machine.
"""

from raspbot_slam.actuators import Actuators
from raspbot_slam.camera import Camera
from raspbot_slam.sensors import Sensors


def connect_bot():
    """Try to construct the Yahboom Raspbot I2C driver.

    Returns:
        A Raspbot driver instance, or None if the vendor library is not
        installed (development machine) or hardware init fails.
    """
    candidates = (
        ("Raspbot_Lib", "Raspbot"),
        ("Raspbot_Lib.Raspbot_Lib", "Raspbot"),
    )
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls()
        except (ImportError, AttributeError):
            continue
        except Exception as exc:  # driver found but hardware init failed
            print(f"[hw] {module_name}.{class_name} import ok but init failed: {exc}")
            return None
    return None


def make_hardware(mock: bool = False):
    """Build (bot, camera, sensors, actuators) as one call.

    Args:
        mock: Force mock mode even if the vendor driver is installed.

    Returns:
        Tuple (bot, camera, sensors, actuators). bot is None in mock mode.
    """
    bot = None if mock else connect_bot()
    camera = Camera()
    sensors = Sensors(bot=bot)
    actuators = Actuators(bot=bot)
    return bot, camera, sensors, actuators
