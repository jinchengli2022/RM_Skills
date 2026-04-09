"""
技能点位采集工具 (Skill Waypoint Recorder)

启动后记录基准位姿，然后实时输出当前位置相对于基准的偏移量，
格式与 skills.py 的 waypoints 完全一致，可直接复制粘贴使用。

用法:
    python src/core/get_skill.py

操作说明:
    - 启动后自动记录当前位姿为基准点（原点）
    - 手动拖动机械臂到目标位置（需先开启拖动示教模式）
    - 实时终端输出相对偏移，格式为 [dx, dy, dz, drx, dry, drz]
    - 按 Enter 键保存当前位置为一个路点
    - 按 q + Enter 退出并打印完整的 waypoints 列表
    - 按 r + Enter 重置基准点为当前位置
    - 按 d + Enter 删除最后一个已保存的路点
"""

import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.get_skill import *

if __name__ == "__main__":
    main()







