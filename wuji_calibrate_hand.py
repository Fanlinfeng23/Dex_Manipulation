#!/usr/bin/env python3
"""Calibrate one Wuji Glove side for a previously selected named SDK user.

The caller selects the SDK user before starting this process.  This wrapper
checks that selection and the physical glove side, then uses Wuji's guided
asynchronous calibration.  It writes a result file only after checking the
SDK's success summary and saved model path.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from wuji_sdk import SdkManager


OFFICIAL_EXAMPLE = Path(os.environ.get(
    "WUJI_CALIBRATION_EXAMPLE",
    str(Path(__file__).resolve().parent / "wuji-sdk/examples/python/wuji_glove/5.calibration.py"),
))
SIDE_NAMES = {"left": "左手", "right": "右手"}


def load_official_terminal_progress():
    """Load the vendor's display without editing or copying its pose guides."""
    spec = importlib.util.spec_from_file_location("wuji_official_calibration", OFFICIAL_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Wuji 官方标定示例：{OFFICIAL_EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TerminalProgress


def side_progress(side: str):
    """Show the target side on every vendor screen and each fallback pose."""
    base = load_official_terminal_progress()

    class SideProgress(base):
        def __init__(self):
            super().__init__()
            self._last_fallback_step = None

        def __call__(self, feedback):
            if not self.enabled:
                step = (
                    feedback.get("step_index"),
                    feedback.get("step_name"),
                    feedback.get("state") in {"done", "waiting_movement"},
                )
                if step != self._last_fallback_step:
                    print(f"当前标定：{SIDE_NAMES[side]}（{side}）", flush=True)
                    self._last_fallback_step = step
            super().__call__(feedback)

        def render(self, feedback, errors):
            super().render(feedback, errors)
            # The vendor clears the terminal at the start of render(), so the
            # side label must be written after its full-screen output.
            print(f"当前标定：{SIDE_NAMES[side]}（{side}）", flush=True)

    return SideProgress()


def model_path(user_id: str, side: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user_id):
        raise ValueError(f"无效的 SDK 用户 ID：{user_id!r}")
    if side not in SIDE_NAMES:
        raise ValueError(f"无效的手侧：{side!r}")
    return Path.home() / ".wuji/sdk/users" / user_id / "models" / f"{side}_hand.urdf"


def validate_current_user(manager, user_id: str) -> None:
    current = manager.current_user()
    if not isinstance(current, Mapping) or current.get("is_default") is not False:
        raise RuntimeError("当前是 Default SDK 用户；标定必须在具名用户下进行")
    if current.get("user_id") != user_id:
        raise RuntimeError(
            f"SDK 当前用户 ID {current.get('user_id')!r} 与指定 ID {user_id!r} 不一致"
        )


def validate_result(result, user_id: str, side: str) -> dict:
    if not isinstance(result, Mapping):
        raise RuntimeError("Wuji SDK 未返回有效的标定结果")
    if result.get("handedness") != side:
        raise RuntimeError(
            f"标定结果手侧 {result.get('handedness')!r} 与指定的 {side!r} 不一致"
        )
    result_user = result.get("sdk_user")
    if not isinstance(result_user, Mapping) or result_user.get("user_id") != user_id:
        raise RuntimeError("标定结果的 SDK 用户 ID 与指定用户不一致")
    if result_user.get("is_default") is not False:
        raise RuntimeError("标定结果意外属于 Default SDK 用户")
    expected = model_path(user_id, side)
    raw_path = result.get("calibrated_urdf")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("标定结果缺少 calibrated_urdf 路径")
    actual = Path(raw_path).expanduser()
    if not actual.is_absolute() or actual != expected:
        raise RuntimeError(f"标定模型路径不符：得到 {actual}，预期 {expected}")
    if not expected.is_file() or expected.stat().st_size == 0:
        raise RuntimeError(f"标定模型不存在或为空：{expected}")
    poses = result.get("poses_collected")
    if isinstance(poses, bool) or not isinstance(poses, int) or poses <= 0:
        raise RuntimeError(f"标定姿势数量无效：{poses!r}")
    return dict(result)


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


async def calibrate_one_side(manager, *, sn: str, side: str, user_id: str, output: Path) -> dict:
    # A user switch inside this process could reload every connected glove's
    # online model.  The parent selects the user before it starts us.
    model_path(user_id, side)
    validate_current_user(manager, user_id)
    glove = manager.connect(sn=sn, device_name=f"calibration_{side}")
    glove_side = str(glove.hand_side().get()).lower()
    if glove_side != side:
        raise RuntimeError(
            f"手套 SN={sn} 是 {glove_side!r}，本次需要 {side!r}；未开始标定"
        )

    progress = side_progress(side)
    print(f"开始标定{SIDE_NAMES[side]} Wuji Glove（SN={sn}，用户={user_id}）", flush=True)
    try:
        progress.start()
        result = await glove.calibrate(
            skip_constraints=False,
            timeout_s=900.0,
            on_feedback=progress,
        )
    finally:
        progress.stop()

    checked = validate_result(result, user_id, side)
    # Never write a success file before all result checks have passed.
    write_result(output, checked)
    print(f"{SIDE_NAMES[side]}标定完成：{checked['calibrated_urdf']}", flush=True)
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wuji Glove 单侧手型标定")
    parser.add_argument("--sn", required=True, help="目标 Wuji Glove 序列号")
    parser.add_argument("--side", required=True, choices=tuple(SIDE_NAMES))
    parser.add_argument("--user-id", required=True, help="父进程已经选中的具名 SDK 用户 ID")
    parser.add_argument("--output", required=True, type=Path, help="成功后写出的 SDK 结果 JSON")
    args = parser.parse_args(argv)
    if not args.sn.strip():
        parser.error("--sn 不能为空")
    manager = None
    status = 1
    try:
        manager = SdkManager.instance()
        asyncio.run(calibrate_one_side(
            manager, sn=args.sn, side=args.side, user_id=args.user_id,
            output=args.output,
        ))
        status = 0
    except KeyboardInterrupt:
        print("\n手型标定已取消", file=sys.stderr, flush=True)
        status = 130
    except Exception as exc:
        print(f"{SIDE_NAMES[args.side]}手型标定失败：{exc}", file=sys.stderr, flush=True)
        status = 1
    finally:
        if manager is not None:
            try:
                manager.disconnect_all()
            except Exception as exc:
                print(f"断开 Wuji 设备失败：{exc}", file=sys.stderr, flush=True)
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
