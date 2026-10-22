# XRoboToolkit Quest capture dependencies

The Quest mode uses the local `native_test.py` capture manager with the official XRoboToolkit Quest v1.0.1 client and PC Service v1.0.0. The release URLs and SHA256 values are listed in `official-assets.json`. Download the two official release assets to `capture/downloads/`, check their SHA256 values, then extract the PC Service package so that `capture/native/opt/apps/roboticsservice/RoboticsServiceProcess` exists. The desktop also needs Android Platform Tools (`adb`) and one authorized Quest.

The official Quest client can be used directly. Its in-headset PC Service address, Head, Controller and Send options must then be set manually. The optional local auto-setup adapter source is under `build-tools/`. To use it, build from the official Quest v1.0.1 APK with the tool versions documented in `build-tools/README.md`, verify the output, install it on the Quest, and create a local `tianji-client.json` from `tianji-client.json.example` with the actual APK SHA256. Keep the generated APK, JSON and any signing keys off Git. The launcher's SHA256 check rejects a mismatched adapter.

The Quest client source is [MIT licensed](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest); the PC Service source is [Apache-2.0 licensed](https://github.com/XR-Robotics/XRoboToolkit-PC-Service). This repository stores neither release binary.
