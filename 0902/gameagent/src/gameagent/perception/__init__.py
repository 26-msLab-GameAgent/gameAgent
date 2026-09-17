"""Frame capture and preprocessing components."""

from gameagent.perception.adb import AdbCaptureAdapter
from gameagent.perception.http import HttpFrameCaptureAdapter
from gameagent.perception.mock import MockCaptureAdapter

__all__ = ["AdbCaptureAdapter", "HttpFrameCaptureAdapter", "MockCaptureAdapter"]

