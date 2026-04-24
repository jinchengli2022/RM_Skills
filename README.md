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
├── handeye_output/
│   └── handeye_result.json              # 手眼标定输出（camera_to_gripper 等）
├── outputs/
│   └── ...                              # 打标签/重建结果（点云、label、metadata）
├── vibe/
│   ├── plan.md
│   └── research.md                      # 过程记录
└── src/
  ├── label.py                         # 采集单帧并生成 label.json
  ├── hand_eye_calibration.py          # 手眼标定（棋盘格/ArUco）
  ├── execute_skill.py                 # 点云配准 + 技能执行入口
  ├── get_skill.py                     # 技能录制入口（封装到 core.get_skill）
  ├── diagnose_grippers.py             # 夹爪诊断脚本
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
  │   ├── zhixing.py                   # 夹爪控制（Modbus/TCP）
  │   └── web_gripper_control.py
  └── Robotic_Arm/
    ├── rm_ctypes_wrap.py            # SDK ctypes 封装
    └── rm_robot_interface.py        # 机械臂底层接口
```

模块关系（主链路）：

- 标定链路：`hand_eye_calibration.py` -> `handeye_output/handeye_result.json`
- 打标签链路：`label.py` + `align/reconstruct.py` -> `outputs/*/point_cloud.ply` + `outputs/*/label.json`
- 执行链路：`execute_skill.py` + `core/skills.py` + `core/demo_project.py` + `core/zhixing.py`
- 录制链路：`get_skill.py` -> `core/get_skill.py` -> 输出技能片段回填 `core/skills.py`

## 核心命令

### 1) 打标签

```bash
python label.py \
  --robot-ip 169.254.128.18 \
  --robot-port 8080 \
  --camera-serial 348522075148 \
  --handeye-result-path ../handeye_output/handeye_result.json \
  --output-dir ../outputs \
  --gripper-width 0.02 \
  --vis
```

注意：当前流程依赖在线标注，可视化过程不清晰，等待改进。

### 2) ArUco 手眼标定

```bash
python -m src.hand_eye_calibration \
      --camera-serial 348522075148 \
    --aruco --marker-size-mm 50 --marker-id 0 \
    --aruco-dict DICT_4X4_50 --auto-move \
    --max-auto-rotation-deg 40 \
    --sample-rotation-step-deg 15
```
注意：默认--auto-move自动标注，ArUco码最好放在视角中心，且给机械臂留下足够的活动裕度。

### 3) 执行技能

```bash
python src/execute_skill.py \
      --robot-ip 169.254.128.18 \
      --robot-port 8080 \
      --skill-name goto_affordance \
      --camera-serial 348522075148 \
      --source-ply /home/rm/ljc/RM_Skills/outputs/kettle_test5/point_cloud.ply \
      --handeye-result-path /home/rm/ljc/RM_Skills/handeye_output/handeye_result.json \
      --vis
```
注意：技能库位于src/core/skill.py文件夹下，技能库格式如下所示：\
speed：机械臂运行极速（0-100）。 \
affordance_pose：操作物体的抓取位姿。技能库中的仅供参考，实际执行execute_skill.py时，外部传入的AFFORDANCE_POSE优先级大于技能库中affordance_pose。 \
ref_pose：参考物体的位姿。用于多物体任务，目前还未有任务明确使用。优先级同'affordance_pose'，外部传入的大于内部。 \
waypoints：相对于affordance_pose的路点（注意是相对坐标）。 \
gripper_positions：0为开，1为闭合。前面的数字表示对应路点所才执行的动作。 
```bash
    "test": {
        "name": "测试",
        "speed": 25,
        "affordance_pose":   [-0.2532, 0.3638, 0.0810, -2.2960, -0.0170, 1.6140],
        "ref_pose":   [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "waypoints": [
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
        ],
        "gripper_positions": {
            0:1,  
            1:0,
            2:1
        },
    },
```

### 4) 录制技能

```bash
python src/get_skill.py
```

## 其他注意事项

- 如果需要单独控制夹爪运行
```bash
python src/core/zhixing.py
```
夹爪如遇不稳定断连情况，在示教器中左侧选择“末端控制” \
切换“工具端电源输出” 24V->0V->24V尝试启动 \
下方出现控制滑条即为启动成功。

