"""Scoped optional timer resolution for the service and its control worker."""
from __future__ import annotations

import ctypes
import logging
import sys


class TimerResolution:
    """Pair each successful timeBeginPeriod with timeEndPeriod in this process.

    On Windows, the default wait quantum can stretch a 4 ms IPC poll or an 11 ms
    control deadline to about 16 ms. A stop-and-wait input stream then spends
    another quantum waiting for its ACK. This requests finer waits, not a
    real-time scheduling guarantee, and never changes registry/power settings.
    """
    def __init__(self):
        self.period_ms = 0
        self._api = None

    def start(self):
        if self.period_ms or sys.platform != 'win32':
            return
        try:
            api = ctypes.WinDLL('winmm')
            api.timeBeginPeriod.argtypes = [ctypes.c_uint]
            api.timeBeginPeriod.restype = ctypes.c_uint
            api.timeEndPeriod.argtypes = [ctypes.c_uint]
            api.timeEndPeriod.restype = ctypes.c_uint
            result = api.timeBeginPeriod(1)
            if result != 0:
                raise OSError(f'timeBeginPeriod returned {result}')
        except OSError as exc:
            logging.getLogger(__name__).warning('High-resolution Windows waits unavailable: %s', exc)
            return
        self._api = api
        self.period_ms = 1

    def close(self):
        if self.period_ms:
            self._api.timeEndPeriod(self.period_ms)
            self.period_ms = 0
            self._api = None
