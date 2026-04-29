# RM_Skills

## 机械臂 IP 说明

- `169.254.128.18`：左臂
- `169.254.128.19`：右臂

## 项目结构（详细）

```text
RM_Skills/
├── README.md
├── requirements.txt
├── setup.py
├── scripts/
│   ├── hand_eye_calibration.py          # 手眼标定入口
│   ├── label.sh                         # Shell 主控的打标签入口
│   ├── get_skill.py                     # 技能录制入口（封装到 core.get_skill）
│   ├── execute_skill.py                 # 旧版点云配准 + 技能执行入口
│   ├── execute_skill.sh                 # 当前推荐的 SAM 分割 + ICP + 技能执行入口
│   └── pick_and_move_pose.py            # 指哪去哪 / 手眼验证
├── handeye_output/
│   └── handeye_result.json              # 手眼标定输出
├── outputs/
│   └── ...                              # 打标签/重建结果（点云、label、metadata）
├── vibe/
│   ├── plan.md
│   └── research.md                      # 过程记录
└── src/
  ├── verify_camera_to_base_with_board.py
  ├── align/
  │   ├── reconstruct.py               # 点云重建与相机设备枚举
  │   ├── ply_registration.py          # 点云配准
  │   ├── tagger.py                    # 标签相关工具
  │   └── test_reconstruct_and_register.py
  ├── core/
  │   ├── demo_project.py              # 机械臂控制封装
  │   ├── skills.py                    # 技能库 SKILL_REGISTRY + SkillExecutor
  │   ├── get_skill.py                 # 路点录制主逻辑
  │   ├── execute_skill_capture.py     # 执行前采集 RGB-D + 点选目标
  │   ├── execute_skill_finalize.py    # 配准、预览、确认后执行技能
  │   ├── label_capture.py             # 打标签采集阶段
  │   ├── label_sam_inference.py       # SAM 分割阶段
  │   ├── label_review.py              # SAM 复核
  │   ├── label_finalize.py            # 点云保存与结果落盘
  │   ├── zhixing.py                   # 夹爪控制（Modbus/TCP）
  │   ├── head_servo_control.py        # 头部相机仰角控制
  │   └── web_gripper_control.py
  └── Robotic_Arm/
    ├── rm_ctypes_wrap.py            # SDK ctypes 封装
    └── rm_robot_interface.py        # 机械臂底层接口
```

模块关系（主链路）：

- 标定链路：`hand_eye_calibration.py` -> `handeye_output/handeye_result.json`
- 打标签链路：`label.sh` -> `src/core/label_*` -> `outputs/*/point_cloud.ply` + `outputs/*/label.json`
- 执行链路：`scripts/execute_skill.sh` -> `src/core/execute_skill_capture.py` -> `src/core/label_sam_inference.py` -> `src/core/execute_skill_finalize.py` -> `core/skills.py`
- 录制链路：`scripts/get_skill.py` -> `src/core/get_skill.py` -> 输出技能片段回填 `core/skills.py`

## 核心命令

### 1) ArUco 手眼标定

```bash
python scripts/hand_eye_calibration.py \
  --mode eye_to_hand \
  --robot-ip 169.254.128.18 \
  --camera-serial 344322073674 \
  --aruco --marker-size-mm 50 --marker-id 0 \
  --aruco-dict DICT_4X4_50 \
  --auto-move \
  --robot-ip 169.254.128.18
```
注意：
- 执行脚本前确认摄像头能看到Aruco码，尽量位于视觉中间。且给机械臂操作留下足够裕度
- `--mode` 必须显式传入，支持 `eye_in_hand` 和 `eye_to_hand`
- `eye_to_hand` 表示头部相机固定、机械夹爪夹持 ArUco 码自动标定
- `--robot-ip` 表示当前执行夹持 ArUco 动作的机械臂，左臂可填 `169.254.128.18`，右臂可填 `169.254.128.19`
- 自动采样的距离、角度限制已改成脚本内部自动判断，不再需要手填
- `eye_to_hand` 的输出会精简为“头部相机相对于机械臂 base 的 4x4 变换矩阵”

### 2) 打Grasp标签

```bash
bash scripts/label.sh \
  --robot-ip 169.254.128.18 \
  --robot-port 8080 \
  --camera-serial 344322073674 \
  --handeye-result-path handeye_output/handeye_result.json \
  --output-dir outputs \
  --gripper-width 0.02
```

注意：
- `344322073674` 表示头部相机
- `label.sh` 是 shell 主控入口，会在内部切换 `arms` 和 `sam` 两个 conda 环境
- 默认 conda 根目录为 `/home/rm/miniconda3`，默认环境名为 `arms` 和 `sam`
- 如果你想手动指定，可以临时设置 `CONDA_BASE`、`ARMS_ENV_NAME`、`SAM_ENV_NAME`
- 运行后会弹出 OpenCV 画面，鼠标单击目标物体，SAM 会自动分割该物体
- 键位说明：`c` 确认当前 mask，`r` 重新点选，`q` 取消退出
- 输出的 `point_cloud.ply` 是目标物体点云，不再是整帧场景点云
- 输出目录还会保存 `mask.png` 和 `sam_overlay.png`
- 标签保存完成后会自动弹出 PyVista 三维预览，显示目标点云、夹爪位置和朝向
- SAM 默认权重路径为 `src/segment-anything/checkpoing/sam_vit_h_4b8939.pth`

### 3) 录制Skill技能库

```bash
python scripts/get_skill.py
```

注意：技能库位于src/core/skills.py文件中，技能库格式如下所示：\
speed：机械臂运行极速（0-100）。 \
affordance_pose：操作物体的抓取位姿。技能库中的仅供参考，实际执行 `execute_skill.sh` 时，外部传入的 AFFORDANCE_POSE 优先级大于技能库中 affordance_pose。 \
ref_pose：参考物体的位姿。用于多物体任务，目前还未有任务明确使用。优先级同'affordance_pose'，外部传入的大于内部。 \
waypoints：相对于affordance_pose的路点（注意是相对坐标；平移单位为米，姿态单位为弧度，例如 `30° = 0.5236`）。 \
gripper_positions：写在某个具体技能条目内部，键为路点索引（从 `0` 开始），值为单夹爪状态；当前项目约定 `0` 为开、`1` 为闭合，并且会在“到达该路点后”执行。 \
录制脚本当前键位：`Enter` 保存路点，`r` 重置 affordance 基准，`d` 删除最后一个路点，`q` 退出输出；若不是交互式 TTY，则会回退成逐行输入。 
```bash
    "goto_affordance": {
        "name": "到达Affordance，然后进行假想倒水任务",
        "description": "先沿 affordance 局部 -Z 方向退后到预备位，再回到 affordance_pose",
        "speed": 20,
        "waypoints": [
            [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.5236],
        ],
        "gripper_positions": {
            0: 0,  # 路点 1 夹爪状态（单夹爪位置）
            1: 1
        },
    },
```

### 4) 执行Grasp+Skill

```bash
bash scripts/execute_skill.sh \
      --robot-ip 169.254.128.18 \
      --robot-port 8080 \
      --skill-name goto_affordance \
      --camera-serial 344322073674 \
      --source-dir outputs/kettle_ljc2 \
      --handeye-result-path /home/rm/ljc/RM_Skills/handeye_output/handeye_result.json \
      --vis
```
说明：
- 执行前会先弹出头部相机画面，单击目标物体后自动调用 SAM 分割
- SAM 复核窗口中，`c` 确认当前 mask，`r` 重新点选，`q` 取消退出
- `--source-dir` 固定读取目录中的 `point_cloud.ply` 和 `label.json`
- `--vis` 会在执行前弹出 PyVista 三维预览，确认配准后的目标点云和夹爪姿态
- `--vis` 打开时，预览关闭后还需要在终端输入 `confirm` 才会真正开始执行技能；`Ctrl+C` 可取消
- 如需保留执行过程中的临时会话目录，可先设置 `KEEP_EXECUTE_SESSION=1`




## 其他注意事项

### 1) 如果需要单独控制夹爪运行
```bash
python src/core/zhixing.py
```
夹爪如遇不稳定断连情况，在示教器中左侧选择“末端控制” \
切换“工具端电源输出” 24V->0V->24V尝试启动 \
下方出现控制滑条即为启动成功。

### 2) 列出所有可控相机
```bash
python3 /home/rm/ljc/RM_Skills/src/align/reconstruct.py --list-cameras
```

### 3) 指哪去哪(可用于验证手眼标定准确度)
```bash
python3 scripts/pick_and_move_pose.py
```

### 4) 控制头部相机仰角
```bash
python src/core/head_servo_control.py
```

### 5) 末端执行器的夹爪坐标
Z轴为朝前
X轴为朝上
Y轴为朝右
