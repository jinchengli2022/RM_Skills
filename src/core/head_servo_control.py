"""Helpers for controlling the two-axis head servo over serial.

This module extracts the control protocol used by
`/home/rm/rmc_aida_l_atom/scripts/head_servo_ctrl.py` and wraps it into a
reusable controller for local scripts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import serial


class HeadServoError(RuntimeError):
    """Raised when head servo communication fails."""


@dataclass(frozen=True)
class HeadServoAngles:
    """Current angle snapshot for the two head servos."""

    pitch: int
    yaw: int


class HeadServoController:
    """Controller for the two head servos used for pitch and yaw."""

    READ_ANGLE_CMD = bytes((0x55, 0x55, 0x05, 0x15, 0x02, 0x01, 0x02))
    WRITE_ANGLE_TEMPLATE = [0x55, 0x55, 0x08, 0x03, 0x01, 0x32, 0x00, 0x00, 0x00, 0x00]

    PITCH_SERVO_ID = 1
    YAW_SERVO_ID = 2

    DEFAULT_CENTER = 500
    DEFAULT_PORT = "/dev/rmUSB3"
    DEFAULT_BAUDRATE = 9600

    def __init__(
        self,
        port: str,
        baudrate: int,
        *,
        timeout: float = 2.0,
        auto_center: bool = False,
        center_pose: tuple[int, int] = (DEFAULT_CENTER, DEFAULT_CENTER),
    ) -> None:
        self._serial = serial.Serial(port, baudrate, timeout=timeout)
        self._lock = threading.Lock()
        self._last_angles: HeadServoAngles | None = None
        if auto_center:
            self.initialize_pose(*center_pose)

    def close(self) -> None:
        """Close the serial connection."""
        if self._serial.is_open:
            self._serial.close()

    def __enter__(self) -> "HeadServoController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _split_u16(value: int) -> tuple[int, int]:
        return value & 0xFF, (value >> 8) & 0xFF

    def _write(self, data: bytes) -> int:
        with self._lock:
            try:
                return self._serial.write(data)
            except serial.SerialException as exc:
                raise HeadServoError(f"Failed to write serial command: {exc}") from exc

    def _read_exact(self, size: int) -> bytes:
        with self._lock:
            try:
                return self._serial.read(size)
            except serial.SerialException as exc:
                raise HeadServoError(f"Failed to read serial response: {exc}") from exc

    def _read_frame(self) -> bytes:
        deadline = time.time() + float(self._serial.timeout or 0)
        buffer = bytearray()

        while time.time() <= deadline:
            chunk = self._read_exact(1)
            if not chunk:
                continue
            buffer.extend(chunk)

            while len(buffer) >= 2 and buffer[0] != 0x55:
                buffer.pop(0)
            if len(buffer) >= 2 and buffer[1] != 0x55:
                buffer.pop(0)
                continue

            if len(buffer) >= 3:
                expected_len = buffer[2] + 2
                while len(buffer) < expected_len and time.time() <= deadline:
                    chunk = self._read_exact(expected_len - len(buffer))
                    if not chunk:
                        break
                    buffer.extend(chunk)
                return bytes(buffer[:expected_len])

        return bytes(buffer)

    def _parse_angle_frame(self, payload: bytes) -> HeadServoAngles | None:
        if len(payload) == 11 and payload[:5] == bytes((0x55, 0x55, 0x09, 0x15, 0x02)):
            angles = HeadServoAngles(
                pitch=(payload[7] << 8) | payload[6],
                yaw=(payload[10] << 8) | payload[9],
            )
            self._last_angles = angles
            return angles

        if len(payload) == 8 and payload[:5] == bytes((0x55, 0x55, 0x06, 0x15, 0x01)):
            servo_id = payload[5]
            angle = (payload[7] << 8) | payload[6]
            cached = self._last_angles or HeadServoAngles(
                pitch=self.DEFAULT_CENTER,
                yaw=self.DEFAULT_CENTER,
            )

            if servo_id == self.PITCH_SERVO_ID:
                angles = HeadServoAngles(pitch=angle, yaw=cached.yaw)
            elif servo_id == self.YAW_SERVO_ID:
                angles = HeadServoAngles(pitch=cached.pitch, yaw=angle)
            else:
                return None

            self._last_angles = angles
            return angles

        return None

    def read_angles(self) -> HeadServoAngles:
        """Read current pitch/yaw angles from the servo controller."""
        last_payload = b""
        for _ in range(3):
            self._write(self.READ_ANGLE_CMD)
            payload = self._read_frame()
            last_payload = payload
            angles = self._parse_angle_frame(payload)
            if angles is not None:
                return angles

        raise HeadServoError(
            "Invalid angle response: "
            f"{last_payload!r}. Check whether the head servo is connected to "
            f"{self._serial.port} with baudrate {self._serial.baudrate}."
        )

    def set_servo_angle(self, servo_id: int, target_angle: int) -> int:
        """Set a raw target angle for a single servo."""
        command = list(self.WRITE_ANGLE_TEMPLATE)
        command[7] = servo_id
        command[8], command[9] = self._split_u16(target_angle)
        return self._write(bytes(command))

    def initialize_pose(self, pitch: int = DEFAULT_CENTER, yaw: int = DEFAULT_CENTER) -> None:
        """Move both joints to the provided initial pose."""
        self.move_pitch_to(pitch)
        time.sleep(0.5)
        self.move_yaw_to(yaw)
        time.sleep(0.5)

    def move_pitch_to(self, target_angle: int) -> int:
        """Move the pitch servo to an absolute target angle."""
        return self.set_servo_angle(self.PITCH_SERVO_ID, int(target_angle))

    def move_yaw_to(self, target_angle: int) -> int:
        """Move the yaw servo to an absolute target angle."""
        return self.set_servo_angle(self.YAW_SERVO_ID, int(target_angle))

    def step_pitch(self, delta: int = 50) -> HeadServoAngles:
        """Step the pitch servo relative to the current angle."""
        current = self.read_angles()
        self.move_pitch_to(current.pitch + int(delta))
        time.sleep(0.2)
        return self.read_angles()

    def step_yaw(self, delta: int = 50) -> HeadServoAngles:
        """Step the yaw servo relative to the current angle."""
        current = self.read_angles()
        self.move_yaw_to(current.yaw + int(delta))
        time.sleep(0.2)
        return self.read_angles()


def read_head_angles(port: str, baudrate: int, *, timeout: float = 2.0) -> HeadServoAngles:
    """Convenience helper for one-shot angle reads."""
    with HeadServoController(port, baudrate, timeout=timeout) as controller:
        return controller.read_angles()


def initialize_head_pose(
    port: str,
    baudrate: int,
    *,
    pitch: int = HeadServoController.DEFAULT_CENTER,
    yaw: int = HeadServoController.DEFAULT_CENTER,
    timeout: float = 2.0,
) -> None:
    """Convenience helper for one-shot head initialization."""
    with HeadServoController(port, baudrate, timeout=timeout) as controller:
        controller.initialize_pose(pitch=pitch, yaw=yaw)


def step_head_pitch(
    port: str,
    baudrate: int,
    *,
    delta: int = 50,
    timeout: float = 2.0,
) -> HeadServoAngles:
    """Convenience helper for one-shot pitch stepping."""
    with HeadServoController(port, baudrate, timeout=timeout) as controller:
        return controller.step_pitch(delta=delta)


def step_head_yaw(
    port: str,
    baudrate: int,
    *,
    delta: int = 50,
    timeout: float = 2.0,
) -> HeadServoAngles:
    """Convenience helper for one-shot yaw stepping."""
    with HeadServoController(port, baudrate, timeout=timeout) as controller:
        return controller.step_yaw(delta=delta)


if __name__ == "__main__":
    port = HeadServoController.DEFAULT_PORT
    baudrate = HeadServoController.DEFAULT_BAUDRATE
    step = 50

    with HeadServoController(port, baudrate, auto_center=True) as controller:
        while True:
            angles = controller.read_angles()
            print(f"Current angles: pitch={angles.pitch}, yaw={angles.yaw}")
            time.sleep(1)

            # print("Pitch up")
            # angles = controller.step_pitch(step)
            # print(f"Now angles: pitch={angles.pitch}, yaw={angles.yaw}")
            # time.sleep(1)

            # print("Pitch down")
            # angles = controller.step_pitch(-step)
            # print(f"Now angles: pitch={angles.pitch}, yaw={angles.yaw}")
            # time.sleep(1)

            # print("Yaw left")
            # angles = controller.step_yaw(step)
            # print(f"Now angles: pitch={angles.pitch}, yaw={angles.yaw}")
            # time.sleep(1)

            # print("Yaw right")
            # angles = controller.step_yaw(-step)
            # print(f"Now angles: pitch={angles.pitch}, yaw={angles.yaw}")
            # time.sleep(1)

            print("Back to center")
            controller.initialize_pose(pitch=400, yaw=500)
            time.sleep(2)
