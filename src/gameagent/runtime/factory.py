"""Build adapters from config."""

from __future__ import annotations

from gameagent.agent import ActionValidator, allowed_actions_from_config
from gameagent.clients import HttpModelClient, MockModelClient
from gameagent.control import AdbControlAdapter, HttpControlAdapter, MockControlAdapter
from gameagent.models import AppConfig, CaptureAdapter, ControlAdapter, ModelClient
from gameagent.perception import AdbCaptureAdapter, HttpFrameCaptureAdapter, MockCaptureAdapter
from gameagent.storage import EpisodeLogger


def build_capture(config: AppConfig) -> CaptureAdapter:
    capture = config.capture
    adapter = str(capture.get("adapter", "mock"))
    if adapter == "mock":
        resize = dict(capture.get("resize", {}))
        return MockCaptureAdapter(
            width=int(resize.get("width", capture.get("width", 720))),
            height=int(resize.get("height", capture.get("height", 1280))),
        )
    if adapter == "adb_screencap":
        resize = dict(capture.get("resize", {}))
        return AdbCaptureAdapter(
            device_id=str(capture.get("device_id", "auto")),
            adb_path=str(capture.get("adb_path", "adb")),
            width=_optional_int(resize.get("width")),
            height=_optional_int(resize.get("height")),
            timeout_s=_ms_to_s(capture.get("timeout_ms", 10000)),
            capture_method=str(capture.get("capture_method", "auto")),
            adb_server_socket=_optional_str(capture.get("adb_server_socket")),
        )
    if adapter == "http_frame_stream":
        endpoint = capture.get("endpoint")
        if not endpoint:
            raise ValueError("capture.endpoint is required for http_frame_stream")
        return HttpFrameCaptureAdapter(
            endpoint=str(endpoint),
            timeout_s=_ms_to_s(capture.get("timeout_ms", 10000)),
        )
    raise ValueError(f"Unknown capture adapter: {adapter}")


def build_control(config: AppConfig) -> ControlAdapter:
    control = config.control
    adapter = str(control.get("adapter", "mock"))
    if adapter == "mock":
        return MockControlAdapter()
    if adapter == "adb_touch":
        return AdbControlAdapter(
            device_id=str(control.get("device_id", "auto")),
            adb_path=str(control.get("adb_path", "adb")),
            min_action_interval_ms=int(control.get("min_action_interval_ms", 250)),
            timeout_s=_ms_to_s(control.get("timeout_ms", 10000)),
            adb_server_socket=_optional_str(control.get("adb_server_socket")),
        )
    if adapter == "http_action_bridge":
        endpoint = control.get("endpoint")
        if not endpoint:
            raise ValueError("control.endpoint is required for http_action_bridge")
        return HttpControlAdapter(
            endpoint=str(endpoint),
            timeout_s=_ms_to_s(control.get("timeout_ms", 10000)),
        )
    raise ValueError(f"Unknown control adapter: {adapter}")


def build_model(config: AppConfig) -> ModelClient:
    model = config.model
    provider = str(model.get("provider", "mock"))
    if provider in {"mock", "mock_policy"}:
        return MockModelClient()
    if provider in {"remote_inference", "http"}:
        endpoint = model.get("endpoint")
        if not endpoint:
            raise ValueError("model.endpoint is required for remote_inference")
        return HttpModelClient(
            endpoint=str(endpoint),
            model_name=str(model.get("model", provider)),
            timeout_s=_ms_to_s(model.get("timeout_ms", 30000)),
            max_retries=int(model.get("max_retries", 2)),
        )
    raise ValueError(f"Unknown model provider: {provider}")


def build_validator(config: AppConfig) -> ActionValidator:
    control = config.control
    guards = config.guards
    return ActionValidator(
        allowed_actions=allowed_actions_from_config(control.get("allowed_actions")),
        clamp_coordinates=bool(guards.get("clamp_coordinates", True)),
        max_repeated_action_count=int(guards.get("max_repeated_action_count", 8)),
    )


def build_logger(config: AppConfig) -> EpisodeLogger:
    storage = config.storage
    return EpisodeLogger(
        run_dir=str(storage.get("run_dir", "./runs")),
        save_frames=bool(storage.get("save_frames", True)),
        save_model_raw_response=bool(storage.get("save_model_raw_response", True)),
    )


def _ms_to_s(value: object) -> float:
    return float(value) / 1000


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
