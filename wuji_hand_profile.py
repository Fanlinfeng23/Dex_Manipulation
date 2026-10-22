#!/usr/bin/env python3
"""Select a Wuji hand model before starting the joint hardware service."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from wuji_sdk import SdkManager


WORKSPACE_ROOT = Path(__file__).resolve().parent
CALIBRATION_SCRIPT = WORKSPACE_ROOT / "wuji_calibrate_hand.py"
SIDES = ("left", "right")
SIDE_NAMES = {"left": "左手", "right": "右手"}


def user_model_path(user: dict, side: str = "right") -> Path | None:
    """Return the SDK's current per-user, per-hand model location."""
    if side not in SIDES:
        raise ValueError(f"无效的手侧：{side}")
    if user.get("is_default"):
        return None
    user_id = str(user.get("user_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user_id):
        raise ValueError(f"无效的 SDK 用户 ID：{user_id!r}")
    return Path.home() / ".wuji/sdk/users" / user_id / f"models/{side}_hand.urdf"


def checked_model_path(user: dict, side: str = "right") -> Path | None:
    path = user_model_path(user, side)
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(f"用户 {user['display_name']} 没有{SIDE_NAMES[side]}标定模型：{path}")
    try:
        if ET.parse(path).getroot().tag != "robot":
            raise ValueError("URDF 根节点不是 robot")
    except (ET.ParseError, OSError, ValueError) as exc:
        raise ValueError(f"{SIDE_NAMES[side]}标定模型不可用：{path}（{exc}）") from exc
    return path.resolve()


def calibration_pending_path(user: dict) -> Path:
    path = user_model_path(user)
    if path is None:
        raise ValueError("完整联合遥操需要具名用户的左右手标定；不能使用 Default 内置模型")
    return path.parent / "joint_calibration.pending.json"


def checked_model_paths(user: dict) -> dict[str, Path]:
    pending = calibration_pending_path(user)
    if pending.exists():
        raise ValueError(f"用户 {user['display_name']} 上次双手标定未完成；请用 --calibrate-user 重新完成左右手标定")
    return {side: checked_model_path(user, side) for side in SIDES}


def model_revision(path: Path) -> tuple | None:
    if not path.is_file():
        return None
    info = path.stat()
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def find_user(manager: SdkManager, name_or_id: str) -> dict:
    matches = [
        user for user in manager.list_users()
        if name_or_id in (user.get("user_id"), user.get("display_name"))
    ]
    if len(matches) != 1:
        raise ValueError(f"SDK 用户 {name_or_id!r} {'不存在' if not matches else '存在重名'}")
    return matches[0]


def select_user(manager: SdkManager, user: dict) -> dict:
    """Switch before glove.connect(); the SDK loads the selected user's URDF there."""
    checked_model_paths(user)
    selected = manager.switch_user(user["user_id"])
    if selected.get("user_id") != user.get("user_id"):
        raise RuntimeError("SDK 切换用户后返回的 ID 与选择不一致")
    return selected


def calibrate_user(manager: SdkManager, name: str, left_glove_sn: str, right_glove_sn: str) -> dict:
    name = name.strip()
    if not name or name == "Default":
        raise ValueError("标定需要一个非空的具名用户，不能使用 Default")
    if not left_glove_sn or not right_glove_sn or left_glove_sn == right_glove_sn:
        raise ValueError("双手标定必须指定两只不同的左、右 Wuji Glove 序列号")
    matches = [
        user for user in manager.list_users()
        if name in (user.get("user_id"), user.get("display_name"))
    ]
    if len(matches) > 1:
        raise ValueError(f"SDK 用户 {name!r} 存在重名")
    user = matches[0] if matches else manager.create_user(name)
    if user.get("is_default"):
        raise ValueError("Default 用户不能保存手型标定")
    pending = calibration_pending_path(user)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps({"user_id": user["user_id"], "required_sides": SIDES}), encoding="utf-8")
    previous = manager.current_user()
    try:
        manager.switch_user(user["user_id"])
        for side, glove_sn in (("left", left_glove_sn), ("right", right_glove_sn)):
            model_path = user_model_path(user, side)
            prior_revision = model_revision(model_path)
            print(f"\n开始为 {user['display_name']} 标定{SIDE_NAMES[side]}（Glove={glove_sn}）；"
                  "请用这一侧手按终端姿势提示操作，两侧完成前不会启动天机。", flush=True)
            if sys.stdin.isatty():
                input(f"准备好{SIDE_NAMES[side]}后，按回车开始标定：")
            with tempfile.TemporaryDirectory(prefix="wuji-calibration-") as result_dir:
                result_path = Path(result_dir) / "result.json"
                result = subprocess.run(
                    [sys.executable, str(CALIBRATION_SCRIPT), "--sn", glove_sn,
                     "--side", side, "--user-id", user["user_id"], "--output", str(result_path)],
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"{SIDE_NAMES[side]}手型标定未完成（退出码 {result.returncode}）；联合遥操未启动")
                if not result_path.is_file():
                    raise RuntimeError(f"{SIDE_NAMES[side]}标定没有返回成功结果；联合遥操未启动")
                summary = json.loads(result_path.read_text(encoding="utf-8"))
                if (summary.get("handedness") != side or
                        summary.get("sdk_user", {}).get("user_id") != user["user_id"] or
                        Path(summary.get("calibrated_urdf", "")).resolve() != model_path.resolve()):
                    raise RuntimeError(f"{SIDE_NAMES[side]}标定结果的用户、侧别或模型路径不匹配")
            checked_model_path(user, side)
            if prior_revision is not None and model_revision(model_path) == prior_revision:
                raise RuntimeError(f"本次标定未更新{SIDE_NAMES[side]}模型；请确认连接的是{SIDE_NAMES[side]} Glove")
            print(f"{SIDE_NAMES[side]}标定模型已保存：{model_path}", flush=True)
        pending.unlink()
        checked_model_paths(user)
    finally:
        if previous.get("is_default"):
            manager.switch_to_default_user()
        else:
            manager.switch_user(previous["user_id"])
    return user


def interactive_choice(manager: SdkManager) -> tuple[str, str | None]:
    current_id = manager.current_user().get("user_id")
    users = manager.list_users()
    available = []
    print("\n选择本次联合遥操的 Wuji 双手模型（左手 + 右手）：", flush=True)
    for user in users:
        if user.get("is_default"):
            continue
        try:
            checked_model_paths(user)
        except ValueError as exc:
            print(f"  - {user['display_name']}：{exc}；输入 c 重新标定")
            continue
        available.append(user)
        current = "，当前用户" if user.get("user_id") == current_id else ""
        print(f"  [{len(available)}] {user['display_name']}（左右手均已标定{current}）")
    print("  [c] 新建用户或重新标定已有用户的左右手")
    answer = input("选择编号 / c：").strip().lower()
    if answer == "c":
        name = input("输入要标定的用户名称：").strip()
        return "calibrate", name
    if answer.isdecimal() and 1 <= int(answer) <= len(available):
        return "existing", str(available[int(answer) - 1]["user_id"])
    raise ValueError("未选择有效模型；联合遥操未启动")


def write_selection(path: Path, user: dict) -> None:
    model_paths = checked_model_paths(user)
    result = {
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "is_default": bool(user["is_default"]),
        "model_paths": {side: str(model) for side, model in model_paths.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(result, tmp, ensure_ascii=False)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="选择或依次标定 Wuji 左右手模型")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--user", help="选择已有标定用户（精确名称或 ID）")
    choice.add_argument("--calibrate-user", help="创建或重新标定具名用户")
    choice.add_argument("--default-model", action="store_true", help="旧参数；完整联合模式不接受未标定的 Default")
    parser.add_argument("--left-glove-sn", default="", help="左手标定手套的序列号")
    parser.add_argument("--right-glove-sn", "--glove-sn", default="", help="右手标定手套的序列号")
    parser.add_argument("--output", required=True, type=Path, help="本次模型选择 JSON 文件")
    args = parser.parse_args()
    manager = SdkManager.instance()
    try:
        if args.user:
            mode, value = "existing", args.user
        elif args.calibrate_user:
            mode, value = "calibrate", args.calibrate_user
        elif args.default_model:
            raise ValueError("完整联合遥操需要左右手标定；请使用 --calibrate-user NAME 或选择已有双手模型")
        else:
            if not sys.stdin.isatty():
                raise ValueError("非交互运行需指定 --user 或 --calibrate-user")
            mode, value = interactive_choice(manager)
        if mode == "calibrate":
            user = calibrate_user(manager, value or "", args.left_glove_sn, args.right_glove_sn)
        else:
            user = find_user(manager, value or "")
        write_selection(args.output, user)
        models = checked_model_paths(user)
        print(f"本次 Wuji 双手模型：{user['display_name']}\n"
              f"  左手：{models['left']}\n  右手：{models['right']}", flush=True)
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        print(f"Wuji 模型准备失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
