# 天机双臂 + Wuji 双手联合遥操启动指南

本指南有两套互斥的实机入口：**Quest 双手柄 + 左右 Wuji Glove**，或 **两只 Vive Tracker + 左右 Wuji Glove**。两套都控制天机左右臂和左右 Wuji Hand2；手套负责手指，Quest 手柄或 Vive Tracker 负责双臂。请在同一台现场电脑的交互式终端运行命令，保持急停可触达、机械臂和灵巧手运动范围净空。不要同时启动两套入口。

两套共用同一操作者的左右手模型。`--model-user operator_a` 选择已完成左右手标定的模型；`--calibrate-user operator_a` 依次重标左、右手。**不带这两个参数时，设备识别完成后会在启动命令所在的电脑终端显示模型菜单；在 `选择编号 / c：` 提示处输入小写 `c` 并回车，才进入手套标定。**这里的 `c` 不是头显或 Tracker 按键。看到 `准备好左手后，按回车开始标定：` 后只做左手姿势，随后按提示完成右手；两侧都成功才会继续实机流程。更换操作员或手套佩戴变化时，请选该操作员已有的双侧模型或重新标定两侧。模型由 Wuji SDK 保存于 `~/.wuji/sdk/users/<用户ID>/models/{left,right}_hand.urdf`。

首次执行 `--calibrate-user` 或在菜单输入 `c` 前，还须具备 **Wuji 官方 SDK 的手套标定示例** `examples/python/wuji_glove/5.calibration.py`。启动器默认从项目下 `wuji-sdk/examples/python/wuji_glove/5.calibration.py` 读取；若 Wuji SDK 安装在其他位置，先在同一终端设置 `export WUJI_CALIBRATION_EXAMPLE=/实际路径/5.calibration.py`。公开仓库不包含该官方示例，需使用本机已获授权的 Wuji SDK 安装；已有双手模型且仅用 `--model-user` 时不需要重新运行标定示例。

## 一、Quest 双手柄 + Wuji Glove（原联合方案）

给天机双臂、左右 Hand2 与 Glove 上电。两只 Quest 手柄分别固定在左右手套连接件上。Quest 3S 用 USB 数据线连接电脑并保持唤醒；首次 ADB 授权或系统锁屏提示仍需在头显内处理。正常启动由电脑自动连接 XRoboToolkit PC Service `127.0.0.1`，打开 Head、Controller、Send，关闭 Switch w/ A Button。把头显刚性固定在颈前，外侧相机朝向双手工作区，避免衣领、下巴或连接件挡住手柄。手柄必须留在相机可见范围内。

已标定双手模型时运行：

```bash
cd /home/tianji/桌面/dex
./tianji_quest.sh joint --model-user operator_a
```

首次使用或更换佩戴后，运行 `./tianji_quest.sh joint --calibrate-user operator_a`；也可以运行不带用户参数的 `./tianji_quest.sh joint`，在电脑终端模型菜单输入 `c`。程序先确认左右 Glove、左右 Hand2 和天机臂反馈，再要求现场输入 **`JOINT-MOVE`**，将双臂移到 Quest 联合方案的初始姿态并保持。收到新鲜 Quest 手柄帧、两只手套骨架帧后，让双手处于自然操作位；准备好后在同一终端按一次**空格**，双臂与双手才首次跟随。手套标定期间不会运动天机臂。

终端按键无需回车：**空格**开始或暂停，**A**重新对齐，**B**暂停，**Q/Esc**结束。手柄短时失追会使双臂与双手暂停；首次开始后，追踪重新稳定时原 Quest 联合控制会从实测臂姿态重锚并自动继续。人为暂停、应用失焦、采集会话重启或硬件故障仍需要现场处理；不要用反复按空格越过故障。Quest 原生输入没有逐侧官方光学有效位，网页显示 `tracked` 只是本地数值检查，不能证明手柄持续被相机看到。

常用只读和恢复命令：

```bash
./tianji_quest.sh check       # Quest、天机和四台 Wuji 设备只读检查
./tianji_quest.sh status      # 查看会话状态
./tianji_quest.sh stop        # 终端丢失时停止本入口会话并请求待机
./tianji_quest.sh home        # 只读规划，通过后输入 HOME 才低速复位
```

`home` 对较大关节或 J6/J7 耦合越限会拒绝自动运动并给出详情；此时需由熟悉天机臂的人员使用厂家流程恢复到模型限位内，不要调大软件限值。物理急停触发后，先排除原因并释放急停，再按现场提示运行 `./tianji_quest.sh recover`。复位完成后重新启动 `joint`，另起一段采集记录。Quest 客户端构建、安装与采集服务配置见[Quest 采集说明](capture/README.md)。

## 二、两只 Vive Tracker + Wuji Glove（新联合方案）

左腕 Tracker 为 `LHR-8180F66F`，右腕为 `LHR-00037B1A`；两台指定基站为 `LHB-BDC4BE18` 和 `LHB-8DCC9AA0`。给两台基站和两只 Tracker 供电，启动 SteamVR，把 Tracker **按真实使用方式刚性固定在左右手背连接件**，露出圆弧传感器。两台基站先前实测中心距约 **0.240 m**，因此示例命令使用 `--expected-baseline-m 0.240`。17:14 的一次只读启动采样中，SteamVR 报 **0.277 m**，虽然双腕各 270/270 帧有效，基站几何仍与实测值不一致而拒绝连接机械臂。遇到此提示须重新量黑色面中心的实际距离、核对 SteamVR 场地解算，再用同一实测值完成标定和启动；不要只为通过门槛把参数改成错误读数。移动基站后重新标定。两站靠得很近会使视角重叠，不能保证整个双手工作区无遮挡。当前无头显 SteamVR Null Driver 的 Chaperone `201` 可在双腕及双站实际有效时使用。

先完成**只读双腕标定**。它不连接或运动天机臂，左右手套也不驱动灵巧手：

```bash
cd /home/tianji/桌面/dex
./tianji_vive.sh calibrate --expected-baseline-m 0.240
```

程序先验证双腕和双站连续定位，再依次提示左腕、右腕 pivot：把前臂稳固地支在桌面或支架上，尽量固定解剖手腕中心，按回车后在 8 秒内沿至少两个方向缓慢转动手腕，勿移动前臂或让连接件滑动。**左腕先前出现 RMS 20.3 mm、最大 40.5 mm，尚未达到实机标定标准**；程序保留 8 mm RMS、20 mm 最大残差门槛，失败时保存 `logs/vive-pivot-*.json` 并允许只重采该腕，不会放松门槛或保存半成品。随后按提示做双腕静止中位与两次 20–30 cm 前向平移，建立操作朝向。成功后生成 `vive_tianji_calibration.json`；改动连接件、基站位置或 SteamVR 站立参考系后必须重标。

如果手腕转心难以保持不动，也可以先用量具或连接件 CAD **分别实测**从每只 HTC Tracker 3.0 安装螺孔中心与贴合环面交点到该侧解剖手腕中心、在 Tracker **局部 X/Y/Z 坐标系**下的带符号向量（单位 mm）。确认 HTC 官方坐标轴方向和连接件安装朝向后，运行：

```bash
./tianji_vive.sh calibrate --expected-baseline-m 0.240 \
  --left-offset-mm 左X 左Y 左Z --right-offset-mm 右X 右Y 右Z
```

也可只给已实测的一侧参数，另一侧继续 pivot。**当前两侧偏移尚未实测，不得把占位文字直接运行，也不要填零或照抄 Wuji 特定安装件的偏移。**程序只检查手工测量值的格式与物理范围，无法从追踪数据证明其绝对精度。标定文件记录每侧方法、向量和 Tracker 序列号。标定成功后可选运行无机器人预演 `./tianji_vive.sh preview --side both --scale 1.0 --seconds 45 --expected-baseline-m 0.240`；预演终端按 **A** 对齐、空格开始、**Q** 退出。

双腕标定文件存在后，启动**双臂 + 双手实机联合遥操**：

```bash
./tianji_vive_joint.sh --model-user operator_a --expected-baseline-m 0.240
```

新操作员改用 `./tianji_vive_joint.sh --calibrate-user operator_a --expected-baseline-m 0.240`；不带用户参数时同样可在电脑终端模型菜单输入 `c`。可加 `--seconds 60` 让测试 60 秒后自动结束，或用 `--scale 0.5` 缩小手腕位移映射；默认比例为与 Wuji 官方米制 TF 一致的 **1:1**。姿态变化始终为 1:1，`--scale` 只影响平移。启动器只读核对四台 Wuji 设备、同一人的左右手模型和两侧新鲜骨架，再进入 Vive 双臂实机入口。它**不会自动复位双臂**，以启动时的实测臂姿态为起点；现场确认运动范围、线缆和急停后，在终端输入 **`LIVE-BOTH`** 并回车。连接完成后按 **A** 以当前双腕与双臂姿态对齐；全部准备好，再按一次**空格**首次开始。按 **B** 暂停、**Q/Esc**退出。

Vive 运行中两腕使用同一次 OpenVR 采样；任一腕失去有效 6DoF 时双臂保持、双手门控关闭。短时失位恢复会重锚；持续失位或姿态跳变后须连续稳定约 0.6 秒，控制层才从**实测臂姿态**重新对齐并自动继续，不把丢踪期间的手部位移累积成机械臂跳变。主动暂停、SteamVR 参考系重置、基站位置变化或机器人故障不会自动恢复。基站**参考位姿**偶尔缺一台但双腕仍为有效 6DoF 时继续跟随；参考位姿重新出现后仍与标定指纹比对。手指控制只有在双臂有效跟随且两只手套持续提供新鲜骨架时才打开；手部通道故障会结束联合会话。

当前双臂实机代码采用 Wuji 官方关节阻抗 K/D、**60%** SDK 速度/加速度比例、Hand2 **0.95 kg / 工具 Z+90 mm** 载荷参数和 **120 Hz** 输入/控制目标频率。仍保留关节/腕部耦合、工作空间、失追暂停、跟踪误差、急停及故障保护；软件另设末端平移 **12 cm/s**、姿态 **0.5 rad/s**、关节目标 **20°/s** 上限。双臂从当前姿态启动、无需先自动回初始位。上述阻抗配置和载荷写入已通过离线测试，**尚未完成现场负载保持及实机运动验收**；120 Hz 是配置目标，实际有效频率以本次日志为准。旧的 30 秒工作区扫动只获得约 **86.6% 双腕同时有效**、左腕最长约 **2.9 秒**失位；代码可暂停和恢复，但不能用软件恢复被遮挡时缺失的真实位置，也不能据静止抖动宣称末端毫米级绝对精度。具体架构、测量和与 Wuji 官方方案的差异见[Vive 双追踪器架构与精度报告](Vive双追踪器遥操架构与精度报告.md)。

实机会话日志保存在 `logs/vive-live-*/` 与 `logs/vive-joint-*/`。结束后可用 `python3 -m vive_tianji.analyze_session logs/vive-live-本次目录/stream.jsonl` 统计双腕失位、输入延迟及目标与实测关节正解的内部误差。如遇关节越限，先退出本会话，再按上一节的 `./tianji_quest.sh home` 流程只读规划与复位；复位后重新执行双腕对齐并另起采集记录。**当前尚未生成正式双腕标定文件，也尚未对本方案做实机运动测试；必须先完成上述标定和现场启动步骤。**
