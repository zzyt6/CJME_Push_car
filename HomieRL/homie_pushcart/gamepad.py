"""Xbox controller through Windows XInput (ctypes, no extra packages)."""
from __future__ import annotations

import ctypes
import sys

BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y = 0x1000, 0x2000, 0x4000, 0x8000
BUTTON_START, BUTTON_BACK = 0x0010, 0x0020
ERROR_DEVICE_NOT_CONNECTED = 1167


class _XInputGamepadStruct(ctypes.Structure):
    _fields_ = [("wButtons", ctypes.c_ushort), ("bLeftTrigger", ctypes.c_ubyte), ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short), ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short), ("sThumbRY", ctypes.c_short)]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", _XInputGamepadStruct)]


class XInputGamepad:
    def __init__(self, index: int = 0, dead_zone: float = 0.12):
        self.index, self.dead_zone = index, dead_zone
        self._lib = None
        if sys.platform == "win32":
            for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
                try:
                    self._lib = ctypes.windll.LoadLibrary(name)
                    break
                except OSError:
                    continue
        self._state = _XInputState()
        self._prev_buttons = 0
        self.connected = False

    @property
    def available(self) -> bool:
        return self._lib is not None

    def poll(self) -> dict:
        """Sticks in [-1, 1] (dead zone applied, up/right positive); buttons pressed this poll in `pressed`."""
        result = {"lx": 0., "ly": 0., "rx": 0., "ry": 0., "buttons": 0, "pressed": 0, "connected": False}
        if self._lib is None:
            return result
        code = self._lib.XInputGetState(self.index, ctypes.byref(self._state))
        if code != 0:
            self.connected = False
            self._prev_buttons = 0
            return result
        g = self._state.Gamepad

        def axis(v: int) -> float:
            x = max(-1.0, min(1.0, v / 32767.0))
            if abs(x) < self.dead_zone:
                return 0.0
            return (abs(x) - self.dead_zone) / (1.0 - self.dead_zone) * (1 if x > 0 else -1)

        result.update(lx=axis(g.sThumbLX), ly=axis(g.sThumbLY), rx=axis(g.sThumbRX), ry=axis(g.sThumbRY),
                      buttons=g.wButtons, pressed=g.wButtons & ~self._prev_buttons, connected=True)
        self._prev_buttons = g.wButtons
        self.connected = True
        return result
