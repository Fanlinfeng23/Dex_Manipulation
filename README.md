# Dex Manipulation：天机双臂与 Wuji 双手联合遥操

本仓库有两个互斥入口。[最终启动指南](Quest天机遥操启动指南.md)给出两套流程的命令、左右手标定、暂停与复位步骤；[双 Vive Tracker 架构与精度报告](Vive双追踪器遥操架构与精度报告.md)说明两追踪器方案与 Wuji 官方方案的关系、已测追踪质量及当前限制。

| 方案 | 臂输入 | 手指输入 | 启动入口 |
| --- | --- | --- | --- |
| Quest 联合版 | XRoboToolkit Quest 双手柄 | 左右 Wuji Glove → 左右 Hand2 | `./tianji_quest.sh joint --model-user operator_a` |
| Vive 联合版 | 左右腕各一只 Vive Tracker 3.0 | 左右 Wuji Glove → 左右 Hand2 | `./tianji_vive.sh calibrate --expected-baseline-m 0.240`，再 `./tianji_vive_joint.sh --model-user operator_a` |

将 `operator_a` 换成实际操作员名称。没有左右手模型时，按最终指南使用 `--calibrate-user`，或在现场终端的模型菜单输入 `c`。Vive 双腕标定只读取 SteamVR，不会运动机器人；Vive 联合版从当前实测双臂姿态对齐，不自动复位。该方案默认腕部平移和姿态均为 1:1；`--scale` 只调整平移比例。Quest 联合版按其现场确认流程进入初始姿态。

**当前验收状态：** Quest 联合版现有代码保留；Vive 与双 Wuji 手的联合监督、门控已完成离线测试。现场左腕 pivot 曾出现 20.3 mm RMS，正式双腕标定文件尚未生成，因此 Vive 联合版尚未完成实机运动验收。离线测试结果不能代替末端精度验收。

## 部署依赖

`./setup_python.sh` 创建两套入口所需的 Python 环境。实际运行还需安装天机 Marvin 厂商 SDK、Wuji SDK 与设备、SteamVR/OpenVR，以及 Quest 模式所需的 XRoboToolkit 官方客户端和 PC Service。手套新标定使用 Wuji 官方 SDK 的 `examples/python/wuji_glove/5.calibration.py`：默认放在本仓库 `wuji-sdk/` 下，或设置 `WUJI_CALIBRATION_EXAMPLE` 指向本机已获授权的示例文件。厂商 SDK、该官方示例、APK、手部模型、标定文件、设备密钥和现场日志均未纳入本仓库。获取方式、默认路径及 `TIANJI_SDK_ROOT`、`TIANJI_QUEST_CAPTURE_DIR`、`TIANJI_ARM_PYTHON` 等覆盖变量见 [发布清单](PUBLISH_MANIFEST.md) 和 [Quest 采集说明](capture/README.md)。

Quest 状态页还需要在 `apps/quest_native_arm_teleop/` 运行 `npm ci --ignore-scripts`，安装 `three` 静态资源。浏览器自动化测试依赖 Node.js 18 或更高版本；`node_modules/` 不随仓库发布。

这是按现场硬件参数配置的源码。其他机器使用前须核对机器人网络、模型、工具几何、Tracker 侧别和运动限位；本仓库中的代码或离线测试不表示任何新设备已通过实机验收。本机旧部署记录只保留在操作员电脑，不纳入公开仓库。
