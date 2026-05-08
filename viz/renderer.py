"""
3D renderer for the UAV static stage scene.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Tuple
from queue import Empty, Queue
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

from sim.environment import EnvironmentConfig
from sim.uav import UAV
from sim.global_state_manager import GlobalStateManager

# Prefer YaHei for CJK rendering; English UI text is used as fallback-friendly default.
plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class SceneRenderer3D:
    """Render UAVs in a 3D mplot3d scene."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        arm_length_m: float = 0.75,
        camera_fov_deg: float = 84.0,
        max_uav_height_m: float = 100.0,
        circle_samples: int = 48,
        cone_theta_samples: int = 20,
        cone_height_samples: int = 10,
    ) -> None:
        self.env_cfg = env_cfg
        self.arm_length_m = arm_length_m
        self.camera_fov_deg = camera_fov_deg
        self.max_uav_height_m = max_uav_height_m
        self.circle_samples = circle_samples
        self.cone_theta_samples = cone_theta_samples
        self.cone_height_samples = cone_height_samples

    def _search_radius_from_height(self, height_m: float) -> float:
        """
        Unified FOV radius model used by rendering and state manager expectations.
        R_fov = H * 1.5
        """
        clamped_h = float(np.clip(height_m, 0.0, self.max_uav_height_m))
        return clamped_h * 1.5

    def _step_towards_target_no_overshoot(
        self,
        uav: UAV,
        target: np.ndarray,
        dt: float,
    ) -> bool:
        """
        Smooth point-to-point motion with acceleration limits and no overshoot.
        Returns True when target is reached.
        """
        target_vec = np.asarray(target, dtype=float).reshape(3)
        to_target = target_vec - uav.position
        dist = float(np.linalg.norm(to_target))
        if dist < 0.08:
            uav.position = target_vec.copy()
            uav.velocity = np.zeros(3, dtype=float)
            return True

        direction = to_target / dist
        current_speed_along = float(np.dot(uav.velocity, direction))
        max_speed = float(uav.max_speed)
        max_acc = float(uav.max_accel)

        # Braking-limited speed profile: v <= sqrt(2 a d)
        braking_limited_speed = float(np.sqrt(max(0.0, 2.0 * max_acc * dist)))
        desired_speed = min(max_speed, braking_limited_speed)

        if current_speed_along < desired_speed:
            next_speed = min(desired_speed, current_speed_along + max_acc * dt)
        else:
            next_speed = max(desired_speed, current_speed_along - max_acc * dt)

        step = min(dist, max(0.0, next_speed) * dt)
        if step <= 1e-9:
            uav.velocity = np.zeros(3, dtype=float)
            return False

        uav.position = uav.position + direction * step
        uav.velocity = direction * (step / dt)

        remaining = float(np.linalg.norm(target_vec - uav.position))
        if remaining < 0.08:
            uav.position = target_vec.copy()
            uav.velocity = np.zeros(3, dtype=float)
            return True
        return False

    def _draw_search_circle(self, ax, uav: UAV, color: str) -> float:
        """Draw camera footprint on ground (z=0) as a circle."""
        radius = self._search_radius_from_height(uav.z)
        theta = np.linspace(0.0, 2.0 * np.pi, self.circle_samples)
        x_circle = uav.x + radius * np.cos(theta)
        y_circle = uav.y + radius * np.sin(theta)
        z_ground = np.full_like(theta, self.env_cfg.z_min)

        ax.plot(x_circle, y_circle, z_ground, color=color, linestyle="--", linewidth=1.7, alpha=0.9)
        ax.plot([uav.x, uav.x], [uav.y, uav.y], [self.env_cfg.z_min, uav.z], color=color, linewidth=1.2, alpha=0.45)
        return radius

    def _draw_camera_cone(self, ax, uav: UAV, radius: float, color: str) -> None:
        """Draw a camera frustum cone clipped at ground (never below z_min)."""
        if uav.z <= self.env_cfg.z_min + 1e-9 or radius <= 1e-9:
            return

        theta = np.linspace(0.0, 2.0 * np.pi, self.cone_theta_samples)
        t = np.linspace(0.0, 1.0, self.cone_height_samples)  # apex->base interpolation
        theta_grid, t_grid = np.meshgrid(theta, t)

        r_grid = t_grid * radius
        x_grid = uav.x + r_grid * np.cos(theta_grid)
        y_grid = uav.y + r_grid * np.sin(theta_grid)
        z_grid = uav.z - t_grid * (uav.z - self.env_cfg.z_min)
        z_grid = np.maximum(z_grid, self.env_cfg.z_min)

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color=color,
            alpha=0.14,
            linewidth=0,
            shade=False,
        )

    def _draw_uav_cross(self, ax, uav: UAV, color: str) -> None:
        """
        Draw one UAV as:
        - two crossed line segments (X-shape in XY plane),
        - one center scatter marker,
        - one text label.
        """
        x, y, z = uav.x, uav.y, uav.z
        l = self.arm_length_m

        # Cross arm-1: from (x-l, y-l) to (x+l, y+l)
        ax.plot([x - l, x + l], [y - l, y + l], [z, z], color=color, linewidth=2.2)
        # Cross arm-2: from (x-l, y+l) to (x+l, y-l)
        ax.plot([x - l, x + l], [y + l, y - l], [z, z], color=color, linewidth=2.2)

        # Central body marker
        ax.scatter([x], [y], [z], color=color, s=70, depthshade=True)

        # Single-line label to avoid stacked-text overlap.
        label_dx = 1.0
        label_dy = 0.9
        ax.text(
            x + label_dx,
            y + label_dy,
            z + 0.35,
            f"{uav.uav_id} | h={z:.1f}m",
            color=color,
            fontsize=9.5,
            weight="bold",
        )

    def _compute_focus_bounds(
        self,
        uavs: List[UAV],
        search_radii: List[float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Compute compact bounds centered around all UAVs.
        This keeps the scene readable even if world size is very large.
        """
        x_low_candidates = np.array([u.x - r for u, r in zip(uavs, search_radii)], dtype=float)
        x_high_candidates = np.array([u.x + r for u, r in zip(uavs, search_radii)], dtype=float)
        y_low_candidates = np.array([u.y - r for u, r in zip(uavs, search_radii)], dtype=float)
        y_high_candidates = np.array([u.y + r for u, r in zip(uavs, search_radii)], dtype=float)
        z_vals = np.array([u.z for u in uavs], dtype=float)

        min_xy_span = 20.0
        min_z_span = 8.0
        xy_margin = 6.0
        z_margin_bottom = 1.0
        z_margin_top = 4.0

        x_center = float((np.min(x_low_candidates) + np.max(x_high_candidates)) / 2.0)
        y_center = float((np.min(y_low_candidates) + np.max(y_high_candidates)) / 2.0)
        z_center = float(np.mean(z_vals))

        x_span = max(float(np.max(x_high_candidates) - np.min(x_low_candidates)) + 2 * xy_margin, min_xy_span)
        y_span = max(float(np.max(y_high_candidates) - np.min(y_low_candidates)) + 2 * xy_margin, min_xy_span)
        z_span = max(float(np.max(z_vals) - np.min(z_vals)) + z_margin_bottom + z_margin_top, min_z_span)

        x_low = max(self.env_cfg.x_min, x_center - x_span / 2)
        x_high = min(self.env_cfg.x_max, x_center + x_span / 2)
        y_low = max(self.env_cfg.y_min, y_center - y_span / 2)
        y_high = min(self.env_cfg.y_max, y_center + y_span / 2)
        z_low = max(self.env_cfg.z_min, z_center - z_span / 2)
        z_high = min(self.env_cfg.z_max, z_center + z_span / 2)

        return (x_low, x_high), (y_low, y_high), (z_low, z_high)

    def render_static_scene(
        self,
        uavs: Iterable[UAV],
        colors: Iterable[str],
        title: str = "Stage-1: Static Multi-UAV 3D Scene",
    ) -> None:
        """Render a single static frame and block on window display."""
        uav_list: List[UAV] = list(uavs)
        color_list: List[str] = list(colors)
        if len(uav_list) != len(color_list):
            raise ValueError("uavs and colors must have the same length.")

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        search_radii: List[float] = [self._search_radius_from_height(uav.z) for uav in uav_list]

        # Compact focus bounds around active UAVs.
        (x_low, x_high), (y_low, y_high), (z_low, z_high) = self._compute_focus_bounds(uav_list, search_radii)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.set_zlim(z_low, z_high)
        ax.set_xlabel("X (m)", labelpad=10)
        ax.set_ylabel("Y (m)", labelpad=10)
        ax.set_zlabel("Height (m)", labelpad=10)
        ax.set_title(title, pad=14)

        # Keep local proportions consistent with current focus bounds.
        ax.set_box_aspect((max(x_high - x_low, 1.0), max(y_high - y_low, 1.0), max(z_high - z_low, 1.0)))

        # Default to top-down view for easier tracking.
        ax.view_init(elev=90, azim=-90)

        # Light grid for spatial awareness.
        ax.grid(True, linestyle="--", alpha=0.4)

        # Draw each UAV.
        for uav, color in zip(uav_list, color_list):
            radius = self._draw_search_circle(ax, uav, color)
            self._draw_camera_cone(ax, uav, radius, color)
            self._draw_uav_cross(ax, uav, color)

        plt.tight_layout()
        plt.show()

    def render_takeoff_landing_animation(
        self,
        uavs: Iterable[UAV],
        colors: Iterable[str],
        target_uav_id: str,
        peak_height_m: float = 100.0,
        steps_per_phase: int = 80,
        interval_ms: int = 60,
        title: str = "UAV Takeoff/Landing Camera Coverage",
    ) -> None:
        """Animate one UAV: ground -> peak height -> ground."""
        uav_list: List[UAV] = list(uavs)
        color_list: List[str] = list(colors)
        if len(uav_list) != len(color_list):
            raise ValueError("uavs and colors must have the same length.")

        target_idx = next((i for i, u in enumerate(uav_list) if u.uav_id == target_uav_id), None)
        if target_idx is None:
            raise ValueError(f"target_uav_id '{target_uav_id}' not found.")

        peak_height = float(np.clip(peak_height_m, self.env_cfg.z_min, self.max_uav_height_m))
        up = np.linspace(self.env_cfg.z_min, peak_height, steps_per_phase)
        down = np.linspace(peak_height, self.env_cfg.z_min, steps_per_phase)
        height_profile = np.concatenate([up, down[1:]])

        # Pre-compute fixed axis bounds using the peak state to avoid per-frame autoscaling jitter/cost.
        peak_state_uavs = [UAV(u.uav_id, np.array([u.x, u.y, u.z])) for u in uav_list]
        peak_state_uavs[target_idx].position[2] = peak_height
        peak_search_radii: List[float] = [self._search_radius_from_height(u.z) for u in peak_state_uavs]
        fixed_bounds = self._compute_focus_bounds(peak_state_uavs, peak_search_radii)
        (fx_low, fx_high), (fy_low, fy_high), (fz_low, fz_high) = fixed_bounds

        fig = plt.figure(figsize=(15, 7))
        ax_top = fig.add_subplot(121, projection="3d")
        ax_side = fig.add_subplot(122, projection="3d")

        def _setup_axis(ax, *, elev: float, azim: float, proj_type: str, view_name: str) -> None:
            ax.set_xlim(fx_low, fx_high)
            ax.set_ylim(fy_low, fy_high)
            ax.set_zlim(fz_low, fz_high)
            ax.set_xlabel("X (m)", labelpad=10)
            ax.set_ylabel("Y (m)", labelpad=10)
            ax.set_zlabel("Height (m)", labelpad=10)
            ax.set_title(view_name, pad=12)
            ax.set_box_aspect((max(fx_high - fx_low, 1.0), max(fy_high - fy_low, 1.0), max(fz_high - fz_low, 1.0)))
            ax.view_init(elev=elev, azim=azim)
            ax.set_proj_type(proj_type)
            ax.grid(True, linestyle="--", alpha=0.4)

        _setup_axis(ax_top, elev=90.0, azim=-90.0, proj_type="ortho", view_name="Top View (Ortho)")
        _setup_axis(ax_side, elev=22.0, azim=-35.0, proj_type="persp", view_name="Side Perspective")

        # Draw static UAVs once (all except target UAV).
        static_indices = [i for i in range(len(uav_list)) if i != target_idx]
        for idx in static_indices:
            uav = uav_list[idx]
            color = color_list[idx]
            radius_top = self._draw_search_circle(ax_top, uav, color)
            _ = radius_top
            self._draw_uav_cross(ax_top, uav, color)

            radius_side = self._draw_search_circle(ax_side, uav, color)
            self._draw_camera_cone(ax_side, uav, radius_side, color)
            self._draw_uav_cross(ax_side, uav, color)

        # Keep dynamic artists in lists so we can remove/re-draw only target UAV.
        dynamic_artists_top: List = []
        dynamic_artists_side: List = []

        def _remove_dynamic(artists: List) -> None:
            for artist in artists:
                try:
                    artist.remove()
                except ValueError:
                    pass
            artists.clear()

        def _draw_dynamic_target(ax, target_uav: UAV, color: str, draw_cone: bool) -> List:
            artists: List = []
            radius = self._search_radius_from_height(target_uav.z)
            theta = np.linspace(0.0, 2.0 * np.pi, self.circle_samples)
            x_circle = target_uav.x + radius * np.cos(theta)
            y_circle = target_uav.y + radius * np.sin(theta)
            z_ground = np.full_like(theta, self.env_cfg.z_min)
            line_circle = ax.plot(x_circle, y_circle, z_ground, color=color, linestyle="--", linewidth=1.7, alpha=0.9)[0]
            artists.append(line_circle)

            line_vertical = ax.plot(
                [target_uav.x, target_uav.x],
                [target_uav.y, target_uav.y],
                [self.env_cfg.z_min, target_uav.z],
                color=color,
                linewidth=1.2,
                alpha=0.45,
            )[0]
            artists.append(line_vertical)


            if draw_cone and target_uav.z > self.env_cfg.z_min + 1e-9 and radius > 1e-9:
                theta_c = np.linspace(0.0, 2.0 * np.pi, self.cone_theta_samples)
                t = np.linspace(0.0, 1.0, self.cone_height_samples)
                theta_grid, t_grid = np.meshgrid(theta_c, t)
                r_grid = t_grid * radius
                x_grid = target_uav.x + r_grid * np.cos(theta_grid)
                y_grid = target_uav.y + r_grid * np.sin(theta_grid)
                z_grid = target_uav.z - t_grid * (target_uav.z - self.env_cfg.z_min)
                z_grid = np.maximum(z_grid, self.env_cfg.z_min)
                cone = ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.14, linewidth=0, shade=False)
                artists.append(cone)

            x, y, z = target_uav.x, target_uav.y, target_uav.z
            l = self.arm_length_m
            arm1 = ax.plot([x - l, x + l], [y - l, y + l], [z, z], color=color, linewidth=2.2)[0]
            arm2 = ax.plot([x - l, x + l], [y + l, y - l], [z, z], color=color, linewidth=2.2)[0]
            body = ax.scatter([x], [y], [z], color=color, s=70, depthshade=True)
            # Single-line label for readability in both top/side view.
            label = ax.text(
                x + 1.0,
                y + 0.9,
                z + 0.35,
                f"{target_uav.uav_id} | h={z:.1f}m",
                color=color,
                fontsize=9.5,
                weight="bold",
            )
            artists.extend([arm1, arm2, body, label])
            return artists

        fig.suptitle(title, fontsize=13)
        target_color = color_list[target_idx]

        def _update(frame_idx: int):
            uav_list[target_idx].position[2] = float(height_profile[frame_idx])
            _remove_dynamic(dynamic_artists_top)
            _remove_dynamic(dynamic_artists_side)
            dynamic_artists_top.extend(_draw_dynamic_target(ax_top, uav_list[target_idx], target_color, draw_cone=False))
            dynamic_artists_side.extend(_draw_dynamic_target(ax_side, uav_list[target_idx], target_color, draw_cone=True))
            return dynamic_artists_top + dynamic_artists_side

        _ = FuncAnimation(
            fig,
            _update,
            frames=len(height_profile),
            interval=interval_ms,
            blit=False,
            repeat=True,
        )
        plt.tight_layout()
        plt.show()

    def render_interactive_control(
        self,
        uavs: Iterable[UAV],
        colors: Iterable[str],
        interval_ms: int = 33,
        title: str = "Interactive Multi-UAV Control",
        discovery_target: np.ndarray | None = None,
        draw_target: bool = True,
        llm_command_text: str = "Reserved (LLM command)",
        global_state_manager: GlobalStateManager | None = None,
        llm_json_every_n_frames: int = 0,
        llm_json_output_path: str | None = None,
        llm_controller=None,
        verbose_runtime_logs: bool = False,
    ) -> None:
        """
        Real-time control mode.
        Input command in terminal: "<id> <x> <y> <z>"
        Example: "A 520 520 80" or "UAV_B 460 520 30"
        """
        uav_list: List[UAV] = list(uavs)
        color_list: List[str] = list(colors)
        if len(uav_list) != len(color_list):
            raise ValueError("uavs and colors must have the same length.")

        id_to_idx = {u.uav_id.upper(): i for i, u in enumerate(uav_list)}
        aliases = {
            "A": "UAV_A",
            "B": "UAV_B",
            "C": "UAV_C",
            "1": "UAV_A",
            "2": "UAV_B",
            "3": "UAV_C",
        }
        cmd_queue: Queue = Queue()

        def _log(msg: str) -> None:
            # All non-LLM logs suppressed; only LLM events are printed
            pass

        # Interactive control UI initialized (logs suppressed)

        def _input_worker() -> None:
            while True:
                try:
                    raw = input("cmd> ").strip()
                except EOFError:
                    break
                if raw.lower() in {"q", "quit", "exit"}:
                    break
                if not raw:
                    continue
                cmd_queue.put(raw)

        threading.Thread(target=_input_worker, daemon=True).start()
        wall_clock_start_sec = time.monotonic()

        # Fixed display area for smoother rendering.
        fx_low, fx_high = self.env_cfg.x_min, self.env_cfg.x_max
        fy_low, fy_high = self.env_cfg.y_min, self.env_cfg.y_max
        fz_low, fz_high = self.env_cfg.z_min, max(self.env_cfg.z_max, self.max_uav_height_m)

        fig = plt.figure(figsize=(18.8, 7))
        ax_top = fig.add_axes([0.04, 0.16, 0.32, 0.74], projection="3d")
        ax_side = fig.add_axes([0.39, 0.16, 0.31, 0.74], projection="3d")
        ax_status = fig.add_axes([0.79, 0.16, 0.20, 0.74])
        ax_status.axis("off")

        def _setup_axis(ax, *, elev: float, azim: float, proj_type: str, view_name: str) -> None:
            ax.set_xlim(fx_low, fx_high)
            ax.set_ylim(fy_low, fy_high)
            ax.set_zlim(fz_low, fz_high)
            ax.set_xlabel("X (m)", labelpad=10)
            ax.set_ylabel("Y (m)", labelpad=10)
            ax.set_zlabel("Height (m)", labelpad=10)
            ax.set_title(view_name, pad=12)
            ax.set_box_aspect((max(fx_high - fx_low, 1.0), max(fy_high - fy_low, 1.0), max(fz_high - fz_low, 1.0)))
            ax.view_init(elev=elev, azim=azim)
            ax.set_proj_type(proj_type)
            ax.grid(True, linestyle="--", alpha=0.35)

        _setup_axis(ax_top, elev=90.0, azim=-90.0, proj_type="ortho", view_name="Top View (Ortho)")
        _setup_axis(ax_side, elev=22.0, azim=-35.0, proj_type="persp", view_name="Side Perspective")
        # Top view does not need Z-axis text/ticks; hiding avoids overlap with Y-axis text.
        ax_top.set_zlabel("")
        ax_top.set_zticks([])

        target_pos = None if discovery_target is None else np.asarray(discovery_target, dtype=float).reshape(3)
        if target_pos is not None and draw_target:
            ax_top.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], marker="*", s=160, color="gold", edgecolors="black")
            ax_top.text(
                float(target_pos[0]) + 1.2,
                float(target_pos[1]) + 1.2,
                float(target_pos[2]) + 0.2,
                "TARGET",
                color="black",
                fontsize=9,
                weight="bold",
            )
            ax_side.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], marker="*", s=120, color="gold", edgecolors="black")

        fig.suptitle(title, fontsize=13)
        status_panel_text = ax_status.text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )
        status_text = fig.text(
            0.02,
            0.085,
            "Input numeric command: uav,skill,params (example: 1,1,500,500,50)",
            fontsize=9,
            color="black",
        )

        # UI input (works even when terminal stdin is unavailable).
        input_box_ax = fig.add_axes([0.08, 0.02, 0.55, 0.05])
        submit_btn_ax = fig.add_axes([0.65, 0.02, 0.08, 0.05])
        cmd_box = TextBox(input_box_ax, "Command ", initial="")
        cmd_btn = Button(submit_btn_ax, "Send")

        dynamic_artists_top: List = []
        dynamic_artists_side: List = []

        def _remove_dynamic(artists: List) -> None:
            for artist in artists:
                try:
                    artist.remove()
                except ValueError:
                    pass
            artists.clear()

        def _draw_one_uav(ax, uav: UAV, color: str, draw_cone: bool) -> List:
            artists: List = []
            # Spread labels with deterministic offsets to reduce overlap between UAVs.
            label_offsets = {
                "UAV_A": (1.0, 0.9),
                "UAV_B": (1.0, -1.1),
                "UAV_C": (-1.4, 0.9),
            }
            dx, dy = label_offsets.get(uav.uav_id, (1.0, 0.9))
            # Draw planned scan trajectory only on top view.
            if (not draw_cone) and uav.current_skill == "scan_sector" and len(uav.waypoints) > 0:
                scan_points = [uav.position.copy(), *[wp.copy() for wp in uav.waypoints]]
                traj_x = [float(p[0]) for p in scan_points]
                traj_y = [float(p[1]) for p in scan_points]
                traj_z = [self.env_cfg.z_min for _ in scan_points]
                artists.append(
                    ax.plot(
                        traj_x,
                        traj_y,
                        traj_z,
                        color=color,
                        linestyle=":",
                        linewidth=1.4,
                        alpha=0.8,
                    )[0]
                )

            radius = self._search_radius_from_height(uav.z)
            theta = np.linspace(0.0, 2.0 * np.pi, self.circle_samples)
            x_circle = uav.x + radius * np.cos(theta)
            y_circle = uav.y + radius * np.sin(theta)
            z_ground = np.full_like(theta, self.env_cfg.z_min)
            artists.append(ax.plot(x_circle, y_circle, z_ground, color=color, linestyle="--", linewidth=1.3, alpha=0.85)[0])
            artists.append(
                ax.plot([uav.x, uav.x], [uav.y, uav.y], [self.env_cfg.z_min, uav.z], color=color, linewidth=1.0, alpha=0.45)[
                    0
                ]
            )
            if draw_cone and uav.z > self.env_cfg.z_min + 1e-9 and radius > 1e-9:
                theta_c = np.linspace(0.0, 2.0 * np.pi, self.cone_theta_samples)
                t = np.linspace(0.0, 1.0, self.cone_height_samples)
                theta_grid, t_grid = np.meshgrid(theta_c, t)
                r_grid = t_grid * radius
                x_grid = uav.x + r_grid * np.cos(theta_grid)
                y_grid = uav.y + r_grid * np.sin(theta_grid)
                z_grid = uav.z - t_grid * (uav.z - self.env_cfg.z_min)
                z_grid = np.maximum(z_grid, self.env_cfg.z_min)
                artists.append(ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.12, linewidth=0, shade=False))

            x, y, z = uav.x, uav.y, uav.z
            l = self.arm_length_m
            artists.append(ax.plot([x - l, x + l], [y - l, y + l], [z, z], color=color, linewidth=2.0)[0])
            artists.append(ax.plot([x - l, x + l], [y + l, y - l], [z, z], color=color, linewidth=2.0)[0])
            artists.append(ax.scatter([x], [y], [z], color=color, s=55, depthshade=True))
            artists.append(
                ax.text(
                    x + dx,
                    y + dy,
                    z + 0.4,
                    f"{uav.uav_id} | h={z:.1f}m",
                    color=color,
                    fontsize=8.3,
                    weight="bold",
                )
            )
            return artists

        def _draw_grid_coverage_top(ax) -> List:
            artists: List = []
            if global_state_manager is None:
                return artists
            grid_size = float(global_state_manager.grid_size)
            z = self.env_cfg.z_min + 0.01
            for gid, ratio in sorted(global_state_manager.grid_coverage_ratio.items()):
                if ratio <= 0.0:
                    continue
                x_idx = ord(gid[0]) - ord("A")
                y_idx = int(gid[1:]) - 1
                x0 = x_idx * grid_size
                x1 = x0 + grid_size
                y0 = y_idx * grid_size
                y1 = y0 + grid_size
                xg = np.array([[x0, x1], [x0, x1]], dtype=float)
                yg = np.array([[y0, y0], [y1, y1]], dtype=float)
                zg = np.array([[z, z], [z, z]], dtype=float)
                completed = ratio >= global_state_manager.search_complete_threshold
                color = "limegreen" if completed else "gold"
                alpha = 0.10 + 0.35 * min(max(ratio, 0.0), 1.0)
                artists.append(
                    ax.plot_surface(
                        xg,
                        yg,
                        zg,
                        color=color,
                        alpha=alpha,
                        linewidth=0,
                        shade=False,
                    )
                )
                artists.append(
                    ax.text(
                        x0 + 0.5 * grid_size,
                        y0 + 0.5 * grid_size,
                        z + 0.15,
                        f"{ratio:.2f}",
                        color="black",
                        fontsize=6.5,
                        ha="center",
                        va="center",
                    )
                )
            return artists

        def _draw_paint_layer_top(ax) -> List:
            """
            Visualize raw scan paint mask first (brush trail),
            then grid coverage is computed/displayed on top of it.
            """
            artists: List = []
            if global_state_manager is None:
                return artists
            mask = global_state_manager.scan_paint_mask
            if mask.size == 0:
                return artists

            ys, xs = np.where(mask)
            if xs.size == 0:
                return artists

            # Keep rendering lightweight for dense masks.
            max_points = 6000
            if xs.size > max_points:
                step = int(np.ceil(xs.size / max_points))
                xs = xs[::step]
                ys = ys[::step]

            x_world = (xs.astype(float) + 0.5) * global_state_manager.paint_resolution_m
            y_world = (ys.astype(float) + 0.5) * global_state_manager.paint_resolution_m
            z_world = np.full_like(x_world, self.env_cfg.z_min + 0.005)
            artists.append(
                ax.scatter(
                    x_world,
                    y_world,
                    z_world,
                    s=4,
                    marker="s",
                    color="royalblue",
                    alpha=0.22,
                    depthshade=False,
                )
            )
            return artists

        dt = interval_ms / 1000.0
        last_detect_frame: dict[str, int] = {}
        detect_cooldown_frames = max(1, int(1000 / max(interval_ms, 1)))
        last_detected_measurement: dict[str, np.ndarray] = {}
        last_manual_command = "N/A"
        llm_json_path = Path(llm_json_output_path) if llm_json_output_path else None

        def parse_grid_id(grid_str: str) -> tuple[float, float, float, float]:
            s = grid_str.strip().upper()
            if len(s) < 2:
                raise ValueError("grid_id must look like F4.")
            col = s[0]
            row_str = s[1:]
            if col < "A" or col > "J":
                raise ValueError("grid column must be A..J.")
            if not row_str.isdigit():
                raise ValueError("grid row must be 1..10.")
            row = int(row_str)
            if row < 1 or row > 10:
                raise ValueError("grid row must be 1..10.")
            x_idx = ord(col) - ord("A")
            y_idx = row - 1
            x_min = x_idx * 100.0
            x_max = x_min + 100.0
            y_min = y_idx * 100.0
            y_max = y_min + 100.0
            return x_min, x_max, y_min, y_max

        def _apply_command(raw: str) -> None:
            nonlocal last_manual_command
            def _command_error(msg: str, uav_name: str | None = None) -> None:
                final_msg = f"[Command Error] {uav_name}: {msg}" if uav_name else f"[Command Error] {msg}"
                _log(final_msg)
                status_text.set_text(final_msg)

            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 2:
                _command_error("Format error. Use: uav,skill,params")
                return
            last_manual_command = raw
            id_raw = parts[0]
            uav_id = aliases.get(id_raw.upper(), id_raw.upper())
            if uav_id not in id_to_idx:
                _command_error(f"Unknown UAV id '{id_raw}'. Please re-enter.")
                return
            idx = id_to_idx[uav_id]
            try:
                skill_id = int(parts[1])
            except ValueError:
                _command_error("Skill id must be integer. Please re-enter.", uav_id)
                return

            def _parse_floats(items: list[str]) -> list[float] | None:
                try:
                    return [float(x) for x in items]
                except ValueError:
                    return None

            if skill_id == 0:
                if len(parts) != 2:
                    _command_error("Skill 0 format: uav,0", uav_id)
                    return
                uav_list[idx].skill_loiter()
                msg = f"[Skill] {uav_id} -> loiter"
                _log(msg)
                status_text.set_text(msg)
                return

            if skill_id == 1:
                if len(parts) != 5:
                    _command_error("Skill 1 format: uav,1,x,y,z", uav_id)
                    return
                vals = _parse_floats(parts[2:])
                if vals is None:
                    _command_error("Coordinates must be numeric.", uav_id)
                    return
                x_cmd, y_cmd, z_cmd = vals
                x_cmd = float(np.clip(x_cmd, self.env_cfg.x_min, self.env_cfg.x_max))
                y_cmd = float(np.clip(y_cmd, self.env_cfg.y_min, self.env_cfg.y_max))
                z_cmd = float(np.clip(z_cmd, self.env_cfg.z_min, self.max_uav_height_m))
                uav_list[idx].skill_goto(np.array([x_cmd, y_cmd, z_cmd], dtype=float))
                msg = f"[Skill] {uav_id} -> goto({x_cmd:.1f}, {y_cmd:.1f}, {z_cmd:.1f})"
                _log(msg)
                status_text.set_text(msg)
                return

            if skill_id == 2:
                if len(parts) < 4:
                    _command_error("Skill 2 format: uav,2,grid1[,grid2,...],alt (e.g. 2,2,F4,G4,H5,45)", uav_id)
                    return
                grid_tokens = parts[2:-1]
                vals = _parse_floats([parts[-1]])
                if vals is None:
                    _command_error("Altitude must be numeric.", uav_id)
                    return
                alt = vals[0]

                parsed_bounds: list[tuple[float, float, float, float]] = []
                seen_grids: set[str] = set()
                for g in grid_tokens:
                    g_norm = g.strip().upper()
                    if not g_norm:
                        continue
                    if g_norm in seen_grids:
                        continue
                    seen_grids.add(g_norm)
                    try:
                        x_min, x_max, y_min, y_max = parse_grid_id(g_norm)
                    except ValueError as e:
                        _command_error(f"Invalid grid_id '{g}': {e}", uav_id)
                        return
                    x_min = float(np.clip(x_min, self.env_cfg.x_min, self.env_cfg.x_max))
                    x_max = float(np.clip(x_max, self.env_cfg.x_min, self.env_cfg.x_max))
                    y_min = float(np.clip(y_min, self.env_cfg.y_min, self.env_cfg.y_max))
                    y_max = float(np.clip(y_max, self.env_cfg.y_min, self.env_cfg.y_max))
                    parsed_bounds.append((x_min, x_max, y_min, y_max))

                if not parsed_bounds:
                    _command_error("No valid grid ids provided for skill 2.", uav_id)
                    return

                alt = float(np.clip(alt, self.env_cfg.z_min, self.max_uav_height_m))
                uav_list[idx].skill_scan_grids(grid_bounds=parsed_bounds, altitude=alt)
                msg = f"[Skill] {uav_id} -> scan_grids({','.join(sorted(seen_grids))}, alt={alt:.1f})"
                _log(msg)
                status_text.set_text(msg)
                return

            if skill_id == 3:
                if len(parts) != 6:
                    _command_error("Skill 3 format: uav,3,x,y,z,verify_alt", uav_id)
                    return
                vals = _parse_floats(parts[2:])
                if vals is None:
                    _command_error("Verify params must be numeric.", uav_id)
                    return
                x_cmd, y_cmd, z_cmd, verify_alt = vals
                x_cmd = float(np.clip(x_cmd, self.env_cfg.x_min, self.env_cfg.x_max))
                y_cmd = float(np.clip(y_cmd, self.env_cfg.y_min, self.env_cfg.y_max))
                z_cmd = float(np.clip(z_cmd, self.env_cfg.z_min, self.max_uav_height_m))
                verify_alt = float(np.clip(verify_alt, self.env_cfg.z_min, self.max_uav_height_m))
                uav_list[idx].skill_descend_verify(
                    approx_pos=np.array([x_cmd, y_cmd, z_cmd], dtype=float),
                    verify_altitude=verify_alt,
                )
                msg = f"[Skill] {uav_id} -> descend_verify({x_cmd:.1f},{y_cmd:.1f},{verify_alt:.1f})"
                _log(msg)
                status_text.set_text(msg)
                return

            if skill_id == 4:
                if len(parts) != 4:
                    _command_error("Skill 4 format: uav,4,x,y", uav_id)
                    return
                vals = _parse_floats(parts[2:])
                if vals is None:
                    _command_error("Grasp params must be numeric.", uav_id)
                    return
                x_cmd, y_cmd = vals
                x_cmd = float(np.clip(x_cmd, self.env_cfg.x_min, self.env_cfg.x_max))
                y_cmd = float(np.clip(y_cmd, self.env_cfg.y_min, self.env_cfg.y_max))
                uav_list[idx].skill_grasp_payload(np.array([x_cmd, y_cmd], dtype=float))
                msg = (
                    f"[Skill] {uav_id} -> grasp_payload({x_cmd:.1f}, {y_cmd:.1f}) "
                    f"[cruise=50m, hold=5s]"
                )
                _log(msg)
                status_text.set_text(msg)
                return

            if skill_id == 5:
                if len(parts) != 3:
                    _command_error("Skill 5 format: uav,5,cruise_alt", uav_id)
                    return
                vals = _parse_floats(parts[2:])
                if vals is None:
                    _command_error("Cruise altitude must be numeric.", uav_id)
                    return
                cruise_alt = float(np.clip(vals[0], self.env_cfg.z_min, self.max_uav_height_m))
                uav_list[idx].skill_return_base(cruise_altitude=cruise_alt)
                msg = f"[Skill] {uav_id} -> return_base"
                _log(msg)
                status_text.set_text(msg)
                return

            _command_error("Unknown skill id. Use 0..5", uav_id)

        def _on_submit(text: str) -> None:
            raw = text.strip()
            if not raw:
                return
            _apply_command(raw)
            cmd_box.set_val("")
            fig.canvas.draw_idle()

        cmd_box.on_submit(_on_submit)
        cmd_btn.on_clicked(lambda _event: _on_submit(cmd_box.text))

        def _update(_frame_idx: int):
            while True:
                try:
                    raw = cmd_queue.get_nowait()
                except Empty:
                    break
                _apply_command(raw)

            for uav in uav_list:
                uav.update_state(dt=dt)
                uav.position[2] = float(np.clip(uav.position[2], self.env_cfg.z_min, self.max_uav_height_m))

                if target_pos is not None:
                    xy_dist = float(np.linalg.norm(target_pos[:2] - uav.position[:2]))
                    if xy_dist <= uav.fov_radius:
                        prev = last_detect_frame.get(uav.uav_id, -10**9)
                        if _frame_idx - prev >= detect_cooldown_frames:
                            sigma = uav.sensor_noise
                            noisy_xy = target_pos[:2] + np.random.normal(0.0, sigma, 2)
                            _log(
                                f"[DETECT] {uav.uav_id} h={uav.z:.1f}m "
                                f"target~({noisy_xy[0]:.1f}, {noisy_xy[1]:.1f}, {target_pos[2]:.1f}) noise={sigma:.2f}"
                            )
                            last_detected_measurement[uav.uav_id] = np.array(
                                [noisy_xy[0], noisy_xy[1], target_pos[2]],
                                dtype=float,
                            )
                            last_detect_frame[uav.uav_id] = _frame_idx

            if global_state_manager is not None:
                global_state_manager.update(uav_list=uav_list, dt=dt)
                if llm_controller is not None:
                    decision = llm_controller.step(uav_list=uav_list, dt=dt, global_state_manager=global_state_manager)
                    commands = decision.get("commands", [])
                    for cmd in commands:
                        _log(f"[LLM CMD] {cmd}")
                        _apply_command(cmd)

            _remove_dynamic(dynamic_artists_top)
            _remove_dynamic(dynamic_artists_side)
            # Draw raw brush-painted area first for debugging/verification.
            dynamic_artists_top.extend(_draw_paint_layer_top(ax_top))
            dynamic_artists_top.extend(_draw_grid_coverage_top(ax_top))
            for uav, color in zip(uav_list, color_list):
                dynamic_artists_top.extend(_draw_one_uav(ax_top, uav, color, draw_cone=False))
                dynamic_artists_side.extend(_draw_one_uav(ax_side, uav, color, draw_cone=True))

            uav_status_lines = []
            for uav in uav_list:
                uav_status_lines.append(
                    f"{uav.uav_id}\n"
                    f"  P=({uav.x:5.1f},{uav.y:5.1f},{uav.z:4.1f}) "
                    f"V={np.linalg.norm(uav.velocity):4.1f}\n"
                    f"  B={uav.battery:5.1f}%  S={uav.current_skill}"
                )

            if target_pos is None:
                target_true_line = "N/A"
            else:
                target_true_line = f"({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})"

            detect_lines = []
            for uav in uav_list:
                if uav.uav_id in last_detected_measurement:
                    m = last_detected_measurement[uav.uav_id]
                    detect_lines.append(f"{uav.uav_id}: ({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f})")
                else:
                    detect_lines.append(f"{uav.uav_id}: N/A")

            real_elapsed_time_sec = time.monotonic() - wall_clock_start_sec

            panel_text = (
                "=== UAV STATUS ===\n"
                + "\n\n".join(uav_status_lines)
                + "\n\n=== LLM COMMAND (RESERVED) ===\n"
                + f"{llm_command_text}\n"
                + f"Last Manual Cmd: {last_manual_command}\n"
                + "\n=== TARGET TRUE POSITION ===\n"
                + f"{target_true_line}\n"
                + "\n=== SYSTEM DETECTED POSITION ===\n"
                + "\n".join(detect_lines)
            )
            if global_state_manager is not None:
                panel_text += (
                    "\n\n=== LLM JSON KEY FIELDS ===\n"
                    + f"real_elapsed_time_sec: {real_elapsed_time_sec:.1f}\n"
                    + f"sim_elapsed_time_sec: {global_state_manager.elapsed_time_sec:.1f}\n"
                    + f"searched_grids: {len(global_state_manager.searched_grids)}\n"
                    + f"unexplored_grids_count: {len(global_state_manager.unexplored_grids)}\n"
                    + f"known_targets: {len(global_state_manager.known_targets)}\n"
                    + f"active_events_pending: {len(global_state_manager.trigger_events)}"
                )

                if llm_json_every_n_frames > 0 and _frame_idx % llm_json_every_n_frames == 0:
                    llm_json = global_state_manager.get_llm_state_json(uav_list=uav_list)
                    _log(f"\n===== LLM JSON @ frame {_frame_idx} =====")
                    _log(llm_json)
                    if llm_json_path is not None:
                        llm_json_path.parent.mkdir(parents=True, exist_ok=True)
                        with llm_json_path.open("w", encoding="utf-8") as f:
                            f.write(llm_json)
            status_panel_text.set_text(panel_text)

            return dynamic_artists_top + dynamic_artists_side

        _ = FuncAnimation(
            fig,
            _update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

