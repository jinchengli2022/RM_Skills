import json
import socket
import time
import logging

class GripperController:
    def __init__(self, socket_client, device_id=1, port=1):
        """
        夹爪控制器初始化
        :param socket_client: 已建立的socket连接
        :param device_id: Modbus从站地址 (默认: 1)
        :param port: 机械臂端口号 (默认: 1)
        """
        self.socket = socket_client
        self.device_id = device_id
        self.port = port

        # Modbus寄存器地址（十进制）
        self.POSITION_REG = 258  # 位置寄存器起始地址
        self.TRIGGER_REG = 264   # 触发寄存器地址

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GripperController")

    def _send_command(self, command_dict):
        """
        发送命令的核心方法
        :param command_dict: 要发送的命令字典
        :return: 服务器响应
        """
        try:
            # 序列化JSON命令
            json_cmd = json.dumps(command_dict) + '\n'  # 添加换行符作为结束符
            self.logger.debug(f"Sending command: {json_cmd.strip()}")

            # 发送命令
            self.socket.sendall(json_cmd.encode('utf-8'))

            # 等待并接收响应
            response = self.socket.recv(1024).decode()
            self.logger.debug(f"Received response: {response.strip()}")
            return response

        except (socket.error, json.JSONDecodeError) as e:
            self.logger.error(f"Command failed: {str(e)}")
            raise RuntimeError("Command transmission failed") from e

    def _send_gripper_register_command(self,fource_register, data_values):
        """
        发送夹爪控制命令
        :param data_values: 数据值列表，例如 [0, 0, 46, 224]对应0-12000 ，  [0, 0, 0, 0]对应0-0
        """
        # 发送位置命令
        pos_cmd = {
            "command": "write_single_register",
            "port": self.port,
            "address": fource_register,
            "num": 1,  # 写入2个寄存器（4字节）
            "data": data_values,
            "device": self.device_id
        }
        self._send_command(pos_cmd)

        # 发送触发命令
        trigger_cmd = {
            "command": "write_single_register",
            "port": self.port,
            "address": self.TRIGGER_REG,
            "data": 1,
            "device": self.device_id
        }
        self._send_command(trigger_cmd)
    def _send_gripper_command(self, data_values):
        """
        发送夹爪控制命令
        :param data_values: 数据值列表，例如 [0, 0, 46, 224]对应0-12000 ，  [0, 0, 0, 0]对应0-0
        """
        # 发送位置命令
        pos_cmd = {
            "command": "write_registers",
            "port": self.port,
            "address": self.POSITION_REG,
            "num": 2,  # 写入2个寄存器（4字节）
            "data": data_values,
            "device": self.device_id
        }
        self._send_command(pos_cmd)

        # 发送触发命令
        trigger_cmd = {
            "command": "write_single_register",
            "port": self.port,
            "address": self.TRIGGER_REG,
            "data": 1,
            "device": self.device_id
        }
        self._send_command(trigger_cmd)

    def read_gripper_command(self):
        """
        发送夹爪控制命令
        :param data_values: 数据值列表，例如 [0, 0, 46, 224]对应0-12000 ，  [0, 0, 0, 0]对应0-0
        """
        # 发送位置命令
        self.logger.info("Reading gipper force...")
        pos_cmd = {"command":"read_holding_registers","port":1,"address":284,"device":1}
        self._send_command(pos_cmd)


    def close_gripper(self, delay=2.0):
        """关闭夹爪（固定数据格式）"""
        self.logger.info("Closing gripper...")
        self._send_gripper_command([0, 0, 0, 0])  # 固定关闭`数据
        time.sleep(delay)

    def open_gripper(self, delay=2.0):
        """打开夹爪（固定数据格式）"""
        self.logger.info("Opening gripper...")
        self._send_gripper_command([0, 0, 46, 224])  # 固定打开数据
        time.sleep(delay)


    def set_gipper_force(self, force,delay):
        self.logger.info("Setting gipper force...")
        self._send_gripper_register_command(261,force)
        time.sleep(delay)



if __name__ == "__main__":
    # 配置连接参数
    HOST = '169.254.128.18'
    PORT = 8080
    DEVICE_PORT = 1
    DEVICE_ID = 1

    try:
        # 建立TCP连接
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")

        # 初始化Modbus模式
        setup_cmd = {
            "command": "set_modbus_mode",
            "port": DEVICE_PORT,
            "baudrate": 115200,
            "timeout": 3
        }
        client.sendall(json.dumps(setup_cmd).encode('utf-8'))
        print("Modbus mode setup response:", client.recv(1024).decode())

        # 初始化夹爪控制器
        gripper = GripperController(client, device_id=DEVICE_ID, port=DEVICE_PORT)

        gripper.set_gipper_force(5,1)
        time.sleep(2)

        # 读力寄存器
        setup_cmd = {"command":"read_holding_registers","port":1,"address":284,"device":1}
        client.sendall(json.dumps(setup_cmd).encode('utf-8'))
        print("Modbus mode setup response:", client.recv(1024).decode())

        # 写力寄存器
        gripper.set_gipper_force(100, 1)
        time.sleep(1)
        setup_cmd = {"command": "read_holding_registers", "port": 1, "address": 261, "device": 1}
        client.sendall(json.dumps(setup_cmd).encode('utf-8'))
        print("Modbus mode setup response:", client.recv(1024).decode())
        # for i in range(50):

            
            # 执行开合测试
        gripper.open_gripper()  # 打开夹爪
        time.sleep(1)
        gripper.close_gripper()          # 关闭夹爪
        time.sleep(1)

        # 设置夹爪力值
        # gripper.set_gipper_force(100, 1)
        # time.sleep(1)
        # 读夹爪力值
        # setup_cmd = {"command": "read_holding_registers", "port": 1, "address": 261, "device": 1}
        # client.sendall(json.dumps(setup_cmd).encode('utf-8'))
        # print("Modbus mode setup response:", client.recv(1024).decode())

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        client.close()
        print("Connection closed")