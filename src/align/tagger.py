import argparse
import json
import os
from datetime import datetime

import numpy as np

try:
	import open3d as o3d
	import open3d.visualization.gui as gui
	import open3d.visualization.rendering as rendering
except ImportError as exc:
	raise ImportError(
		"open3d is required. Install with: pip install open3d"
	) from exc


def normalize(vec: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(vec)
	if norm < 1e-8:
		return vec.copy()
	return vec / norm


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
	axis = normalize(axis)
	if np.linalg.norm(axis) < 1e-8:
		return np.eye(3)
	x, y, z = axis
	c = np.cos(angle_rad)
	s = np.sin(angle_rad)
	one_c = 1.0 - c
	return np.array(
		[
			[c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
			[y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
			[z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
		],
		dtype=np.float64,
	)


def rotation_from_z(direction: np.ndarray) -> np.ndarray:
	"""Return rotation matrix that maps +Z to direction."""
	src = np.array([0.0, 0.0, 1.0], dtype=np.float64)
	dst = normalize(direction)

	dot_val = float(np.clip(np.dot(src, dst), -1.0, 1.0))
	if abs(dot_val - 1.0) < 1e-8:
		return np.eye(3)
	if abs(dot_val + 1.0) < 1e-8:
		return axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), np.pi)

	axis = np.cross(src, dst)
	angle = float(np.arccos(dot_val))
	return axis_angle_to_matrix(axis, angle)


class GraspTaggerApp:
	def __init__(self, ply_path: str, save_path: str | None = None):
		self.ply_path = os.path.abspath(ply_path)
		self.save_path = self._resolve_save_path(save_path)

		self.pcd = o3d.io.read_point_cloud(self.ply_path)
		if len(self.pcd.points) == 0:
			raise ValueError(f"Point cloud is empty: {self.ply_path}")

		self.points_np = np.asarray(self.pcd.points)
		self.mouse_anchor_position = self.points_np.mean(axis=0)
		self.position_offset = np.zeros(3, dtype=np.float64)
		self.current_position = self.mouse_anchor_position + self.position_offset
		# Gripper local frame in world: columns are local X/Y/Z axes.
		self.current_rotation = np.eye(3, dtype=np.float64)
		self.rotate_step_deg = 6.0

		bounds = self.pcd.get_axis_aligned_bounding_box()
		self.diagonal = np.linalg.norm(bounds.get_extent())
		self.gripper_depth = max(self.diagonal * 0.14, 0.04)
		self.finger_height = max(self.diagonal * 0.02, 0.006)
		self.finger_thickness = max(self.diagonal * 0.012, 0.004)
		self.gripper_width = max(self.diagonal * 0.05, 0.015)
		self.gripper_width_step = max(self.diagonal * 0.01, 0.004)
		self.offset_step = max(self.diagonal * 0.008, 0.002)
		self.gripper_width_min = max(self.diagonal * 0.01, 0.006)
		self.gripper_width_max = max(self.diagonal * 0.18, 0.08)

		self.window = None
		self.scene_widget = None
		self.scene = None

		self.pcd_material = rendering.MaterialRecord()
		self.pcd_material.shader = "defaultUnlit"
		self.pcd_material.point_size = 3.0

		self.gripper_material = rendering.MaterialRecord()
		self.gripper_material.shader = "defaultLit"
		self.gripper_material.base_color = [0.95, 0.2, 0.2, 1.0]

		self.help_label = None
		self.gripper_name = "grasp_gripper"

	@staticmethod
	def _resolve_save_path(save_path: str | None) -> str:
		if save_path:
			return os.path.abspath(save_path)
		return ""

	def run(self) -> None:
		app = gui.Application.instance
		app.initialize()

		self.window = app.create_window("Grasp Tagger", 1280, 800)
		self.window.set_on_layout(self._on_layout)
		self.window.set_on_key(self._on_key)

		self.scene_widget = gui.SceneWidget()
		self.scene_widget.set_on_mouse(self._on_mouse)
		self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
		self.scene = self.scene_widget.scene

		self.scene.add_geometry("point_cloud", self.pcd, self.pcd_material)
		self.scene.set_background([0.08, 0.08, 0.08, 1.0])

		bounds = self.pcd.get_axis_aligned_bounding_box()
		# Open3D camera setup API differs by version.
		if hasattr(self.scene_widget, "setup_camera"):
			self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
		elif hasattr(self.scene, "setup_camera"):
			self.scene.setup_camera(60.0, bounds, bounds.get_center())
		else:
			raise RuntimeError(
				"Open3D camera setup API not found. Please upgrade open3d."
			)

		self.help_label = gui.Label(
			"Mouse Move: anchor point | I/J/K/L: camera-plane offset | N/M: world Z offset | W/S, A/D, Z/X: rotate local X/Y/Z | Up/Down: width | Space: save json+png"
		)

		self.window.add_child(self.scene_widget)
		self.window.add_child(self.help_label)

		self._refresh_arrow()
		app.run()

	def _on_layout(self, layout_context: gui.LayoutContext) -> None:
		content_rect = self.window.content_rect
		self.scene_widget.frame = content_rect

		pref = self.help_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
		self.help_label.frame = gui.Rect(
			content_rect.x + 12,
			content_rect.y + 12,
			pref.width + 12,
			pref.height + 8,
		)

	def _on_mouse(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
		if event.type != gui.MouseEvent.Type.MOVE:
			return gui.Widget.EventCallbackResult.IGNORED

		def depth_cb(depth_image: o3d.geometry.Image) -> None:
			x = event.x - self.scene_widget.frame.x
			y = event.y - self.scene_widget.frame.y

			depth_np = np.asarray(depth_image)
			if x < 0 or y < 0 or y >= depth_np.shape[0] or x >= depth_np.shape[1]:
				return

			depth = float(depth_np[y, x])
			if depth >= 1.0:
				return

			world = self.scene_widget.scene.camera.unproject(
				x,
				y,
				depth,
				self.scene_widget.frame.width,
				self.scene_widget.frame.height,
			)

			def update_on_main() -> None:
				self.mouse_anchor_position = np.asarray(world, dtype=np.float64)
				self._update_current_position()
				self._refresh_arrow()

			gui.Application.instance.post_to_main_thread(self.window, update_on_main)

		self.scene.scene.render_to_depth_image(depth_cb)
		return gui.Widget.EventCallbackResult.HANDLED

	def _on_key(self, event: gui.KeyEvent) -> bool:
		if event.type != gui.KeyEvent.DOWN:
			return False

		step = np.deg2rad(self.rotate_step_deg)
		key = event.key
		local_x_axis = self.current_rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
		local_y_axis = self.current_rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
		local_z_axis = self.current_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

		if key == gui.KeyName.W:
			self._rotate_gripper(axis=local_x_axis, angle_rad=-step)
			return True
		if key == gui.KeyName.S:
			self._rotate_gripper(axis=local_x_axis, angle_rad=step)
			return True
		if key == gui.KeyName.A:
			self._rotate_gripper(axis=local_y_axis, angle_rad=step)
			return True
		if key == gui.KeyName.D:
			self._rotate_gripper(axis=local_y_axis, angle_rad=-step)
			return True
		if key == gui.KeyName.Z:
			self._rotate_gripper(axis=local_z_axis, angle_rad=step)
			return True
		if key == gui.KeyName.X:
			self._rotate_gripper(axis=local_z_axis, angle_rad=-step)
			return True
		if key == gui.KeyName.I:
			self._adjust_offset_camera_plane(delta_right=0.0, delta_up=self.offset_step)
			return True
		if key == gui.KeyName.K:
			self._adjust_offset_camera_plane(delta_right=0.0, delta_up=-self.offset_step)
			return True
		if key == gui.KeyName.J:
			self._adjust_offset_camera_plane(delta_right=-self.offset_step, delta_up=0.0)
			return True
		if key == gui.KeyName.L:
			self._adjust_offset_camera_plane(delta_right=self.offset_step, delta_up=0.0)
			return True
		if key == gui.KeyName.N:
			self._adjust_offset(np.array([0.0, 0.0, self.offset_step], dtype=np.float64))
			return True
		if key == gui.KeyName.M:
			self._adjust_offset(np.array([0.0, 0.0, -self.offset_step], dtype=np.float64))
			return True
		if self._is_up_key(key):
			self._adjust_width(self.gripper_width_step)
			return True
		if self._is_down_key(key):
			self._adjust_width(-self.gripper_width_step)
			return True
		if key == gui.KeyName.SPACE:
			self._save_outputs()
			return True
		return False

	@staticmethod
	def _is_up_key(key: object) -> bool:
		candidates = []
		for name in ("UP", "UP_ARROW", "ARROW_UP"):
			if hasattr(gui.KeyName, name):
				candidates.append(getattr(gui.KeyName, name))
		if key in candidates:
			return True
		# GLFW fallback keycode for ArrowUp.
		return isinstance(key, int) and key == 265

	@staticmethod
	def _is_down_key(key: object) -> bool:
		candidates = []
		for name in ("DOWN", "DOWN_ARROW", "ARROW_DOWN"):
			if hasattr(gui.KeyName, name):
				candidates.append(getattr(gui.KeyName, name))
		if key in candidates:
			return True
		# GLFW fallback keycode for ArrowDown.
		return isinstance(key, int) and key == 264

	def _adjust_width(self, delta: float) -> None:
		self.gripper_width = float(
			np.clip(self.gripper_width + delta, self.gripper_width_min, self.gripper_width_max)
		)
		self._refresh_arrow()

	def _update_current_position(self) -> None:
		self.current_position = self.mouse_anchor_position + self.position_offset

	def _adjust_offset(self, delta_xyz: np.ndarray) -> None:
		self.position_offset = self.position_offset + delta_xyz
		self._update_current_position()
		self._refresh_arrow()

	def _get_camera_plane_axes(self) -> tuple[np.ndarray, np.ndarray]:
		# Fallback to world XY axes if camera basis cannot be queried.
		default_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
		default_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

		if self.scene_widget is None or self.scene_widget.scene is None:
			return default_right, default_up

		camera = self.scene_widget.scene.camera
		if camera is None or not hasattr(camera, "get_view_matrix"):
			return default_right, default_up

		view = np.asarray(camera.get_view_matrix(), dtype=np.float64)
		if view.shape != (4, 4):
			return default_right, default_up

		# For world->camera view matrix, row 0/1 are camera right/up axes in world coordinates.
		right = normalize(view[0, :3])
		up = normalize(view[1, :3])
		if np.linalg.norm(right) < 1e-8 or np.linalg.norm(up) < 1e-8:
			return default_right, default_up

		# Keep the two axes orthonormal for stable keyboard motion.
		up = normalize(up - np.dot(up, right) * right)
		if np.linalg.norm(up) < 1e-8:
			return default_right, default_up
		return right, up

	def _adjust_offset_camera_plane(self, delta_right: float, delta_up: float) -> None:
		right_axis, up_axis = self._get_camera_plane_axes()
		delta = right_axis * delta_right + up_axis * delta_up
		self._adjust_offset(delta)

	def _rotate_gripper(self, axis: np.ndarray, angle_rad: float) -> None:
		rotation = axis_angle_to_matrix(axis, angle_rad)
		self.current_rotation = rotation @ self.current_rotation
		# Keep matrix numerically stable and orthonormal.
		u, _, vh = np.linalg.svd(self.current_rotation)
		self.current_rotation = u @ vh
		self._refresh_arrow()

	@staticmethod
	def _create_box(center: np.ndarray, size_xyz: np.ndarray) -> o3d.geometry.TriangleMesh:
		box = o3d.geometry.TriangleMesh.create_box(
			width=float(size_xyz[0]),
			height=float(size_xyz[1]),
			depth=float(size_xyz[2]),
		)
		box.translate(center - size_xyz / 2.0)
		return box

	def _build_gripper(self) -> o3d.geometry.TriangleMesh:
		# Local frame: +Z is approach direction toward contact; contact center is at z=0.
		finger_len = self.gripper_depth * 0.55
		bridge_len = self.gripper_depth * 0.2
		handle_len = self.gripper_depth * 0.45

		gap = self.gripper_width
		t = self.finger_thickness
		h = self.finger_height

		z_finger_center = -finger_len / 2.0
		left_x = -(gap / 2.0 + t / 2.0)
		right_x = +(gap / 2.0 + t / 2.0)

		left_finger = self._create_box(
			center=np.array([left_x, 0.0, z_finger_center]),
			size_xyz=np.array([t, h, finger_len]),
		)
		right_finger = self._create_box(
			center=np.array([right_x, 0.0, z_finger_center]),
			size_xyz=np.array([t, h, finger_len]),
		)

		bridge_width = gap + 2.0 * t
		z_bridge_center = -finger_len - bridge_len / 2.0
		bridge = self._create_box(
			center=np.array([0.0, 0.0, z_bridge_center]),
			size_xyz=np.array([bridge_width, h * 1.2, bridge_len]),
		)

		handle_radius = max(t * 0.6, 0.002)
		handle = o3d.geometry.TriangleMesh.create_cylinder(
			radius=handle_radius,
			height=handle_len,
			resolution=20,
		)
		handle.compute_vertex_normals()
		# Cylinder default axis is +Z; move it behind the bridge to keep handle outside object.
		handle.translate(np.array([0.0, 0.0, -finger_len - bridge_len - handle_len / 2.0]))

		gripper = left_finger + right_finger
		gripper += bridge
		gripper += handle
		gripper.compute_vertex_normals()

		gripper.rotate(self.current_rotation, center=np.zeros(3))
		# current_position is the contact center between two fingertips.
		gripper.translate(self.current_position)
		return gripper

	def _refresh_arrow(self) -> None:
		if self.scene.has_geometry(self.gripper_name):
			self.scene.remove_geometry(self.gripper_name)

		gripper = self._build_gripper()
		self.scene.add_geometry(self.gripper_name, gripper, self.gripper_material)

	def _save_label_json(self) -> None:
		if not self.save_path:
			base_name = os.path.splitext(os.path.basename(self.ply_path))[0]
			self.save_path = os.path.join(
				os.path.dirname(self.ply_path),
				f"label.json",
			)

		payload = {
			"ply_path": self.ply_path,
			"gripper_contact_center": self.current_position.tolist(),
			"mouse_anchor_position": self.mouse_anchor_position.tolist(),
			"gripper_position_offset": self.position_offset.tolist(),
			"gripper_direction": normalize(
				self.current_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
			).tolist(),
			"gripper_rotation_matrix": self.current_rotation.tolist(),
			"gripper_width": float(self.gripper_width),
			"gripper_depth": float(self.gripper_depth),
			"finger_thickness": float(self.finger_thickness),
			"finger_height": float(self.finger_height),
			# Backward-compatible aliases for earlier field names.
			"arrow_position": self.current_position.tolist(),
			"arrow_direction": normalize(
				self.current_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
			).tolist(),
			"saved_at": datetime.now().isoformat(timespec="seconds"),
		}

		os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
		with open(self.save_path, "w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)

		print(f"Saved grasp label: {self.save_path}")

	def _save_screenshot(self) -> None:
		if not self.save_path:
			return

		if self.scene is None:
			print("Failed to save screenshot: scene is not initialized")
			return

		image_path = os.path.splitext(self.save_path)[0] + ".png"

		def image_cb(image: o3d.geometry.Image) -> None:
			ok = o3d.io.write_image(image_path, image)
			if ok:
				print(f"Saved screenshot: {image_path}")
			else:
				print(f"Failed to save screenshot: {image_path}")

		self.scene.scene.render_to_image(image_cb)

	def _save_outputs(self) -> None:
		self._save_label_json()
		self._save_screenshot()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Interactive PLY grasp point tagger")
	parser.add_argument("ply", type=str, help="Path to input .ply file")
	parser.add_argument(
		"--save",
		type=str,
		default=None,
		help="Path to output json file (default: <ply_name>_grasp_label.json)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	app = GraspTaggerApp(ply_path=args.ply, save_path=args.save)
	app.run()


if __name__ == "__main__":
	main()
