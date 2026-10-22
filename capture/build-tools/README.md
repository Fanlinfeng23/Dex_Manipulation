# Tianji 自动启动适配包

基于 XRoboToolkit Quest 官方 v1.0.1，单独包名 `com.xrobotoolkit.client.quest.tianji`，应用名 **XRoboToolkit Tianji Auto**。原版不卸载、不清数据。构建和验证后，操作者在本机填写上一级 `tianji-client.json`；仓库只提供 `.example` 模板。

仅修改 Android manifest、应用名称/版本和新增的 `AutoSetupActivity.smali`。构建脚本会验证官方 `assets/` 与 `lib/` 文件未变。构建输出使用本地签名，不是官方发布 APK；启动器按本机 `capture/tianji-client.json` 中的 SHA256 核验。此仓库不保存 APK、签名材料或本机批准清单。

电脑通过 `native_test.configure_tianji_client` 向 AndroidProxy 发送唯一探针，收到 Unity 回显后，请求 Activity 一次性开启 Head / Controller / Send 并连接 127.0.0.1。每个应用进程只处理一次，重复调用不反转开关。Switch w/ A Button 保持官方默认关闭。随后用 PC SDK 接收至少三个时钟递增的完整 Head/Controller 数据帧；未通过则机械臂尚未启动。

`rebuild_auto_client.py` 可用本目录工具重建（不安装、不修改正在使用的批准清单）。先按 `capture/README.md` 下载已核验的官方 APK，再自行准备下列构建工具：

- [Apktool v3.0.3](https://github.com/iBotPeaches/Apktool/releases/tag/v3.0.3)
- [uber-apk-signer v1.3.0](https://github.com/patrickfav/uber-apk-signer/releases/tag/v1.3.0)
- [Temurin JRE 17](https://github.com/adoptium/temurin17-binaries/releases/tag/jdk-17.0.20.1%2B1)
- [XRoboToolkit 官方源码 v1.0.1](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest/tree/v1.0.1)

只在电脑正式入口启用自动适配。原始采集工具仍可独立使用官方原版；正式入口可用 `TIANJI_QUEST_CLIENT=official` 临时回退，此时需原版手动设置。切换客户端前须结束会话。系统锁屏、边界提示、首次 ADB 授权仍由操作员处理，不做自动绕过。
