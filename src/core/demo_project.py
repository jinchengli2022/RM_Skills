import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Robotic_Arm.rm_robot_interface import *
# from Robotic_Arm.rm_robot_interface import *
# 定义机械臂型号到点位的映射  
arm_models_to_points = {  
    "RM_65": [  
        [0, 20, 70, 0, 90, 0],
        [0.3, 0, 0.3, 3.14, 0, 0],
        [0.2, 0, 0.3, 3.14, 0, 0],
        [0.3, 0, 0.3, 3.14, 0, 0],
        [0.2, 0.05, 0.3, 3.14, 0, 0],
        [0.2, -0.05, 0.3, 3.14, 0, 0] ,
    ],  
    "RM_75": [  
        [0, 20, 0, 70, 0, 90, 0],
        [0.297557, 0, 0.337061, 3.142, 0, 3.142],
        [0.097557, 0, 0.337061, 3.142, 0, 3.142],
        [0.297557, 0, 0.337061, 3.142, 0, 3.142],
        [0.257557, -0.08, 0.337061, 3.142, 0, 3.142],
        [0.257557, 0.08, 0.337061, 3.142, 0, 3.142],
    ], 
    "RML_63": [  
        [0, 20, 70, 0, 90, 0],
        [0.448968, 0, 0.345083, 3.142, 0, 3.142],
        [0.248968, 0, 0.345083, 3.142, 0, 3.142],
        [0.448968, 0, 0.345083, 3.142, 0, 3.142],
        [0.408968, -0.1, 0.345083, 3.142, 0, 3.142],
        [0.408968, 0.1, 0.345083, 3.142, 0, 3.142]  ,
    ], 
    "ECO_65": [  
        [0, 20, 70, 0, -90, 0],
        [0.352925, -0.058880, 0.327320, 3.141, 0, -1.57],
        [0.152925, -0.058880, 0.327320, 3.141, 0, -1.57],
        [0.352925, -0.058880, 0.327320, 3.141, 0, -1.57],
        [0.302925, -0.158880, 0.327320, 3.141, 0, -1.57],
        [0.302925, 0.058880, 0.327320, 3.141, 0, -1.57],
    ],
    "GEN_72": [  
        [0, 0, 0, -90, 0, 0, 0],
        [0.1, 0, 0.4, 3.14, 0, 0],
        [0.3, 0, 0.4, 3.14, 0, 0],
        [0.3595, 0, 0.4265, 3.142, 0, 0],
        [0.3595, 0.03, 0.4265, 3.142, 0, 0],
        [0.3595, 0.03, 0.4665, 3.142, 0, 0],
    ],
    "ECO_63": [  
        [0, 20, 70, 0, -90, 0],
        [0.544228, -0.058900, 0.468274, 3.142, 0, -1.571],
        [0.344228, -0.058900, 0.468274, 3.142, 0, -1.571],
        [0.544228, -0.058900, 0.468274, 3.142, 0, -1.571],
        [0.504228, -0.108900, 0.468274, 3.142, 0, -1.571],
        [0.504228, -0.008900, 0.468274, 3.142, 0, -1.571],
    ],
} 


class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        """
        Initialize and connect to the robotic arm.

        Args:
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("\nFailed to connect to the robot arm\n")
            exit(1)
        else:
            print(f"\nSuccessfully connected to the robot arm: {self.handle.id}\n")

    def get_arm_model(self):
        """Get robotic arm mode.
        """
        res, model = self.robot.rm_get_robot_info()
        if res == 0:
            return model["arm_model"]
        else:
            print("\nFailed to get robot arm model\n")

    def check_and_clear_errors(self):
        """检查并清除机械臂错误状态"""
        # 获取当前状态
        state = self.robot.rm_get_current_arm_state()
        if state[0] == 0 and 'err' in state[1]:
            err_info = state[1]['err']
            print(f"Current arm state: {err_info}")
            # 如果有错误，尝试清除
            if err_info.get('err_len', 0) > 0:
                self.robot.rm_clear_all_err()
                print("Cleared all errors")

    def disconnect(self):
        """
        Disconnect from the robot arm.

        Returns:
            None
        """
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("\nSuccessfully disconnected from the robot arm\n")
        else:
            print("\nFailed to disconnect from the robot arm\n")

    def get_arm_software_info(self):
        """
        Get the software information of the robotic arm.

        Returns:
            None
        """
        software_info = self.robot.rm_get_arm_software_info()
        if software_info[0] == 0:
            print("\n================== Arm Software Information ==================")
            print("Arm Model: ", software_info[1]['product_version'])
            print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
            print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
            print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
            print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
            print("==============================================================\n")
        else:
            print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")

    def movej(self, joint, v=20, r=0, connect=0, block=1):
        """
        Perform movej motion.

        Args:
            joint (list of float): Joint positions.
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movej_result = self.robot.rm_movej(joint, v, r, connect, block)
        if movej_result == 0:
            print("\nmovej motion succeeded\n")
        else:
            print("\nmovej motion failed, Error code: ", movej_result, "\n")

    def movel(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movel motion.

        Args:
            pose (list of float): End position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("\nmovel motion succeeded\n")
        else:
            print("\nmovel motion failed, Error code: ", movel_result, "\n")

    def movec(self, pose_via, pose_to, v=20, r=0, loop=0, connect=0, block=1):
        """
        Perform movec motion.

        Args:
            pose_via (list of float): Via position [x, y, z, rx, ry, rz].
            pose_to (list of float): End position for the circular path [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            loop (int, optional): Number of loops. Defaults to 0.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movec_result = self.robot.rm_movec(pose_via, pose_to, v, r, loop, connect, block)
        if movec_result == 0:
            print("\nmovec motion succeeded\n")
        else:
            print("\nmovec motion failed, Error code: ", movec_result, "\n")

    def movej_p(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movej_p motion.

        Args:
            pose (list of float): Position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("\nmovej_p motion succeeded\n")
        else:
            print("\nmovej_p motion failed, Error code: ", movej_p_result, "\n")

    def move_one_point(self, point, v=30, block=1):
        if len(point) != 6:
            print("point err!")
            return

        print(point[2])
        # 创建一个新的位置列表,z轴坐标增加0.15米
        point1 = point.copy()
        point1[2] = point[2] + 0.15

        # 移动到目标点上方
        self.movel(point1, v=v, block=block)
        
        # 下降到目标点
        self.movel(point, v=v, block=block)
        
        # 查询夹爪当前状态
        ret, gripper_state = self.robot.rm_get_gripper_state()
        if ret == 0:
            print(f"Before pick - Gripper position: {gripper_state.get('actpos')}")
        
        # 使用位置控制闭合夹爪（非阻塞模式）
        print("Closing gripper to pick object...")
        import time
        
        # 发送闭合命令（非阻塞）
        ret = self.robot.rm_set_gripper_position(200, False, 0)
        print(f"Gripper close command sent, result: {ret}")
        
        # 手动等待夹爪闭合
        print("Waiting for gripper to close...")
        time.sleep(2.0)
        
        # 查询夹爪状态确认
        ret, gripper_state = self.robot.rm_get_gripper_state()
        if ret == 0:
            print(f"✓ After pick - Gripper position: {gripper_state.get('actpos')}")
        
        # 关键：执行一个小的关节运动来恢复到位设备为关节
        ret, current_state = self.robot.rm_get_current_arm_state()
        if ret == 0 and 'joint' in current_state:
            current_joint = current_state['joint']
            self.robot.rm_movej(current_joint, 10, 0, 0, 1)
            print("✓ Restored current device to joint after gripper pick")
        
        # 上升
        self.movel(point1, v=v, block=block)
        
        # 创建一个新的位置列表,y轴坐标增加0.10米
        point2 = point1.copy()
        point2[1] = point1[1] + 0.10
        self.movel(point2, v=v, block=block)

    def check_gripper_config(self):
        """检查夹爪配置和状态"""
        print("\n=== Checking Gripper Configuration ===")
        
        # 1. 检查末端生态协议模式
        ret, mode = self.robot.rm_get_rm_plus_mode()
        print(f"End effector protocol mode: {mode} (ret: {ret})")
        if mode == 0:
            print("⚠️  Warning: Protocol is disabled! Enabling it...")
            # 尝试启用协议（常用波特率115200或256000）
            self.robot.rm_set_rm_plus_mode(115200)
            import time
            time.sleep(0.5)
            ret, mode = self.robot.rm_get_rm_plus_mode()
            print(f"After enabling - mode: {mode}")
        
        # 2. 查询夹爪状态
        ret, gripper_state = self.robot.rm_get_gripper_state()
        if ret == 0:
            print(f"Gripper state: {gripper_state}")
            print(f"  - Enable: {gripper_state.get('enable_state')}")
            print(f"  - Status: {gripper_state.get('status')}")
            print(f"  - Position: {gripper_state.get('actpos')}")
            print(f"  - Error: {gripper_state.get('error')}")
        else:
            print(f"⚠️  Failed to get gripper state, error: {ret}")
        
        # 3. 设置夹爪行程（可选，如果失败也没关系）
        print("Setting gripper route (optional)...")
        ret = self.robot.rm_set_gripper_route(0, 1000)
        if ret == 0:
            print(f"✓ Set gripper route successfully")
        else:
            print(f"⚠️  Set gripper route failed ({ret}), using default range")
        
        print("=== Gripper Check Complete ===\n")

    def test_gripper_basic(self):
        """基础夹爪测试 - 使用位置控制（非阻塞模式）"""
        print("\n=== Testing Gripper with Position Control (Non-blocking) ===")
        import time
        
        # 测试1：完全张开
        print("Test 1: Opening gripper to max position (1000)...")
        ret = self.robot.rm_set_gripper_position(1000, False, 0)
        print(f"Command sent, result: {ret}")
        time.sleep(2.0)  # 等待运动完成
        ret, state = self.robot.rm_get_gripper_state()
        if ret == 0:
            print(f"✓ Current position: {state.get('actpos')}")
        
        # 测试2：完全闭合
        print("\nTest 2: Closing gripper to min position (0)...")
        ret = self.robot.rm_set_gripper_position(0, False, 0)
        print(f"Command sent, result: {ret}")
        time.sleep(2.0)
        ret, state = self.robot.rm_get_gripper_state()
        if ret == 0:
            print(f"✓ Current position: {state.get('actpos')}")
        
        # 测试3：半开
        print("\nTest 3: Setting gripper to middle position (500)...")
        ret = self.robot.rm_set_gripper_position(500, False, 0)
        print(f"Command sent, result: {ret}")
        time.sleep(2.0)
        ret, state = self.robot.rm_get_gripper_state()
        if ret == 0:
            print(f"✓ Current position: {state.get('actpos')}")
        
        print("=== Gripper Position Control Test Complete ===\n")

    def init_robot(self):
        # 移动到初始位置
        self.movel([0.133367, 0.418547, 0.45754, 2.444, -1.329, 2.506], v=30, block=1)
        
        # 检查夹爪配置
        self.check_gripper_config()
        
        # 使用位置控制打开夹爪（非阻塞模式）
        print("Opening gripper using position control...")
        import time
        
        # 关键：使用非阻塞模式，避免卡死
        ret = self.robot.rm_set_gripper_position(1000, False, 0)
        print(f"Gripper open command sent, result: {ret}")
        
        # 手动等待，给夹爪时间运动
        print("Waiting for gripper to open...")
        time.sleep(2.0)  # 等待2秒让夹爪运动
        
        # 查询夹爪状态确认
        # ret, state = self.robot.rm_get_gripper_state()
        # if ret == 0:
        #     print(f"✓ Gripper position after open: {state.get('actpos')}")
        
        # 关键：执行一个小的关节运动来恢复到位设备为关节
        # ret, current_state = self.robot.rm_get_current_arm_state()
        # if ret == 0 and 'joint' in current_state:
        #     current_joint = current_state['joint']
        #     self.robot.rm_movej(current_joint, 10, 0, 0, 1)
        #     print("✓ Restored current device to joint")


def main():
    # Create a robot arm controller instance and connect to the robot arm
    robot_controller = RobotArmController("169.254.128.19", 8080, 3)
    # robot_controller.set_gripper_pick_on(500, 200)j家爪
    # Get API version
    print("\nAPI Version: ", rm_api_version(), "\n")

    # Get basic arm information
    robot_controller.get_arm_software_info()

    arm_model = robot_controller.get_arm_model()
    points = arm_models_to_points.get(arm_model, [])

    # 注意：如果需要使用拖动示教，请注释掉下面的运动控制代码
    # print(robot_controller.robot.rm_start_drag_teach(0))
    
    # 确保拖动示教模式已关闭，否则运动控制指令会失败
    # robot_controller.robot.rm_stop_drag_teach()
    
    # 检查并清除可能的错误状态
    # robot_controller.check_and_clear_errors()
    
    ##################### test demo ####################
    print("\n=== Step 1: Initialize robot ===")
    robot_controller.init_robot()
    
    
    print("\n=== Step 2: Execute pick and place ===")
    p = [0.052932, 0.33543, -0.033416, -3.102, -0.007, -1.463]
    robot_controller.move_one_point(p)

    # while(True):
    #     print(robot_controller.robot.rm_get_current_arm_state())

    robot_controller.disconnect()


if __name__ == "__main__":
    main()
