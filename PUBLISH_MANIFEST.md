# Publication manifest

This is a curated source tree for the two Tianji teleoperation modes. It is deliberately separate from the upstream Wuji repository and the original operator workstation.

## Included

- Root launchers, Wuji bilateral hand selection/calibration/control scripts, Vive tracking and dual-arm integration modules, read-only diagnostic tools, tests and operator documentation.
- `apps/quest_native_arm_teleop/`: local dual-arm server, kinematics, control process, native bridge, web status page, configuration and tests. Its original Python/JavaScript source is included; generated runtime data is not.
- `capture/`: local XRoboToolkit Quest capture manager and source for the optional auto-setup Android adapter. No APK or redistributed PC Service binary is included.
- `licenses/Wuji-MIT.txt`: upstream Wuji copyright and MIT terms for the Wuji-derived integration context.

## Intentionally excluded

- Device-specific calibration files and per-operator Wuji hand models; session logs, pose records, photographs and videos.
- All virtual environments, Node modules, caches and generated artifacts.
- Tianji Marvin control and kinematics SDK; MANUS SDK; Wuji SDK checkout; PC Service binaries; Quest APKs; APK signing tools and signing keys. Obtain these from their respective rights holders.

## External dependencies and paths

| Dependency | Default local placement | Override |
| --- | --- | --- |
| Tianji Marvin SDK | `./TJ_FX_ROBOT_CONTRL_SDK` | `TIANJI_SDK_ROOT` |
| XRoboToolkit PC Service extraction | `./capture/native/opt/apps/roboticsservice` | `TIANJI_XROBOT_SERVICE_DIR` |
| Quest capture manager | `./capture` | `TIANJI_QUEST_CAPTURE_DIR` |
| Arm Python environment | `./apps/quest_native_arm_teleop/.venv-linux` | `TIANJI_ARM_PYTHON` |
| Wuji Python environment | `./.venv` | `TIANJI_HAND_PYTHON` |
| Wuji official glove calibration example | `./wuji-sdk/examples/python/wuji_glove/5.calibration.py` | `WUJI_CALIBRATION_EXAMPLE` |
| OpenVR package (if separate environment) | `./.vive-venv` | `TIANJI_OPENVR_SITE` |

The Quest status server serves `three` from `apps/quest_native_arm_teleop/node_modules/`. Run `npm ci --ignore-scripts` in that app directory after checkout; browser automation needs Node.js 18 or later. `node_modules/` remains excluded from the repository.

Set `TIANJI_ROBOT_IP`, `TIANJI_DEVICE_INTERFACE` and `TIANJI_HOST_CIDR` when the controller and dedicated network use different addresses. The local defaults describe the original workstation.

For a new operator hand calibration, supply the Wuji official `5.calibration.py` from an authorized local Wuji SDK installation at the default path above, or point `WUJI_CALIBRATION_EXAMPLE` to its actual file. The wrapper imports its terminal pose guide during calibration. This upstream example is not redistributed here. Existing left/right hand models selected with `--model-user` do not need that guide at runtime.

Quest auto-setup also needs a locally built and installed APK plus a locally created `capture/tianji-client.json` with its SHA256. Without it, the unmodified official client remains available and needs setup in the headset. See `capture/README.md`.

The code contains installation-specific private network addresses and hardware serial mappings used as defaults. They are not passwords or cryptographic keys; review and override them before use on a different robot. Do not commit local `.env` files, secrets, hand models, logs, APKs or vendor SDK binaries.

## Third-party sources

- [Wuji hand teleoperation](https://github.com/wuji-technology/wuji-hand-teleop) is MIT licensed; copyright notice and terms are copied in `licenses/Wuji-MIT.txt`.
- [XRoboToolkit Quest client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest) is MIT licensed; its license is copied in `licenses/XRoboToolkit-Quest-MIT.txt`. This repository links to its official APK and keeps local adapter source, not the APK.
- [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) is Apache-2.0 licensed; its license is copied in `licenses/XRoboToolkit-PC-Service-Apache-2.0.txt`. Download/extract the official release separately.
- Tianji Marvin SDK and MANUS SDK redistribution terms were not established; neither is included.
