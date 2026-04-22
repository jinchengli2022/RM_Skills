"""Web-based gripper control wrapper for RealMan teach pendant APIs.

This module wraps the same HTTP endpoints used by the web UI under
"扩展 -> 末端控制 -> 夹爪", so external scripts can call them directly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class WebGripperError(RuntimeError):
    """Raised when web gripper API call fails."""


_NO_PROXY_OPENER = build_opener(ProxyHandler({}))
_LOCAL_USERS = {
    "root": {"password": "realman", "permissions": ["admin"]},
    "admin": {"password": "123", "permissions": []},
    "user": {"password": "123", "permissions": []},
}


def _resolve_token(token: str | None = None) -> str | None:
    return token or os.environ.get("REALMAN_TOKEN")


def login_and_get_token(
    username: str,
    password: str,
    *,
    base_url: str = "http://169.254.128.18:8090",
    timeout: float = 2.0,
) -> str:
    """Resolve a session token using the teach pendant's actual login flow.

    The web frontend validates username/password locally, then calls
    /arm/getArmSoftwareInfo and reads token_v from the response.
    """
    account = _LOCAL_USERS.get(username)
    if account is None or account["password"] != password:
        raise WebGripperError("Invalid username or password.")

    url = f"{base_url.rstrip('/')}/arm/getArmSoftwareInfo"
    data = _post_json(url, payload={}, timeout=timeout, token=None)
    token = data.get("data", {}).get("token_v")
    if not token:
        raise WebGripperError(f"Token not found in login response: {data}")
    return token


def _post_json(
    url: str,
    payload: Dict[str, Any] | None = None,
    timeout: float = 2.0,
    token: str | None = None,
) -> Dict[str, Any]:
    body = json.dumps(payload or {}).encode("utf-8")
    headers = {
        "Content-Type": "application/json;charset=UTF-8;",
        "Accept": "application/json, text/plain, */*",
    }
    resolved_token = _resolve_token(token)
    if resolved_token:
        headers["token"] = resolved_token

    request = Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with _NO_PROXY_OPENER.open(request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise WebGripperError(f"HTTP error {exc.code} for {url}: {exc}") from exc
    except URLError as exc:
        raise WebGripperError(f"Network error for {url}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WebGripperError(f"Invalid JSON response from {url}: {raw[:200]}") from exc

    if data.get("code") != 0:
        raise WebGripperError(f"API returned error for {url}: {data}")
    return data


def get_gripper_state(
    base_url: str = "http://169.254.128.18:8090",
    timeout: float = 2.0,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> Dict[str, Any]:
    """Read gripper state from web API endpoint /arm/getEepsStateInfo."""
    url = f"{base_url.rstrip('/')}/arm/getEepsStateInfo"
    resolved_token = _resolve_token(token)
    if not resolved_token and username and password:
        resolved_token = login_and_get_token(username, password, base_url=base_url, timeout=timeout)
    return _post_json(url, payload={}, timeout=timeout, token=resolved_token)


def set_gripper_position_via_web(
    position: int,
    *,
    base_url: str = "http://169.254.128.18:8090",
    dof: int = 1,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
    wait: bool = False,
    wait_timeout: float = 3.0,
    poll_interval: float = 0.1,
    tolerance: int = 2,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """Set gripper target position through teach pendant web backend.

    Args:
        position: Target position, usually in [0, 1000] for one-DOF gripper.
        base_url: Teach pendant backend base URL (default: 169.254.128.18:8090).
        dof: Gripper DOF index, typically 1.
        token: Optional auth token. If omitted, REALMAN_TOKEN env var is used.
        username/password: Optional credentials for auto-login when token is absent.
        wait: If True, poll current position until target is reached.
        wait_timeout: Max seconds to wait for reaching target.
        poll_interval: Polling interval in seconds when wait=True.
        tolerance: Position absolute error tolerance when wait=True.
        timeout: Per-request timeout in seconds.

    Returns:
        API response dict from /arm/setEepsCtrlInfo.

    Raises:
        ValueError: Invalid input range.
        WebGripperError: Communication or API-level failure.
    """
    if not isinstance(position, int):
        raise ValueError("position must be int")
    if not 0 <= position <= 1000:
        raise ValueError("position must be in [0, 1000]")
    if dof <= 0:
        raise ValueError("dof must be positive")

    resolved_token = _resolve_token(token)
    if not resolved_token and username and password:
        resolved_token = login_and_get_token(username, password, base_url=base_url, timeout=timeout)

    control_url = f"{base_url.rstrip('/')}/arm/setEepsCtrlInfo"
    payload = {"type": 1, "dof": dof, "data": [position]}
    result = _post_json(control_url, payload=payload, timeout=timeout, token=resolved_token)

    if not wait:
        return result

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        state = get_gripper_state(
            base_url=base_url,
            timeout=timeout,
            token=resolved_token,
        )
        pos = state.get("state_info", {}).get("pos", [None])
        current = pos[0] if pos else None
        if isinstance(current, (int, float)) and abs(int(current) - position) <= tolerance:
            return result
        time.sleep(poll_interval)

    raise WebGripperError(
        f"Target not reached in {wait_timeout}s (target={position}, tolerance={tolerance})."
    )

def set_left_gripper_position_via_web(position: int, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for left gripper (DOF 1)."""
    return set_gripper_position_via_web(
            position,
            base_url="http://169.254.128.18:8090",
            username=os.environ.get("REALMAN_USER", "root"),
            password=os.environ.get("REALMAN_PASSWORD", "realman"),
            wait=True,
        )

def set_right_gripper_position_via_web(position: int, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for right gripper (DOF 1)."""
    return set_gripper_position_via_web(
            position,
            base_url="http://169.254.128.19:8090",
            username=os.environ.get("REALMAN_USER", "root"),
            password=os.environ.get("REALMAN_PASSWORD", "realman"),
            wait=True,
        )

if __name__ == "__main__":
    for i in range(5):
        set_left_gripper_position_via_web(0)
        set_right_gripper_position_via_web(1000)
        time.sleep(1)
        set_left_gripper_position_via_web(1000)
        set_right_gripper_position_via_web(0)
        time.sleep(1)

    # Example usage:
    # export REALMAN_TOKEN='...'
    # python web_gripper_control.py
    # try:
    #     response = set_gripper_position_via_web(
    #         500,
    #         username=os.environ.get("REALMAN_USER", "root"),
    #         password=os.environ.get("REALMAN_PASSWORD", "realman"),
    #         wait=True,
    #     )
    #     print("Gripper command successful:", response)
    # except WebGripperError as e:
    #     print("Gripper command failed:", e)