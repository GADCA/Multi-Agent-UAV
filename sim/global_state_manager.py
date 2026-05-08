"""
Global state manager for LLM-facing mission state export.
"""

from __future__ import annotations

from collections import deque
import json
import math
from typing import Deque

import numpy as np

from sim.uav import UAV


class GlobalStateManager:
    """
    Maintain global belief state and export LLM-friendly JSON snapshots.
    """

    def __init__(
        self,
        map_width: float = 1000.0,
        map_height: float = 1000.0,
        grid_size: float = 100.0,
        known_truth_targets: list[np.ndarray] | None = None,
    ) -> None:
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.grid_size = float(grid_size)
        self.cols = int(self.map_width // self.grid_size)
        self.rows = int(self.map_height // self.grid_size)
        if self.cols != 10 or self.rows != 10:
            raise ValueError("Current naming scheme requires a 10x10 grid.")

        all_grids = self._build_all_grid_ids()
        self.searched_grids: set[str] = set()
        self.unexplored_grids: set[str] = set(all_grids)
        self.search_complete_threshold: float = 0.95
        self.paint_resolution_m: float = 5.0  # brush canvas resolution
        self.paint_width = int(self.map_width // self.paint_resolution_m)
        self.paint_height = int(self.map_height // self.paint_resolution_m)
        self.scan_paint_mask = np.zeros((self.paint_height, self.paint_width), dtype=bool)

        self.grid_coverage_ratio: dict[str, float] = {gid: 0.0 for gid in all_grids}
        self._prev_positions: dict[str, np.ndarray] = {}
        self.trigger_events: Deque[dict] = deque()
        self.elapsed_time_sec: float = 0.0

        self.known_truth_targets = [np.asarray(t, dtype=float).reshape(3) for t in (known_truth_targets or [])]
        self.known_targets: list[dict] = []
        self._known_target_keys: set[tuple[float, float, float, str]] = set()
        self._low_battery_uavs: set[str] = set()
        self._latest_target_estimate: dict | None = None

    def _build_all_grid_ids(self) -> list[str]:
        grid_ids: list[str] = []
        for yi in range(10):
            for xi in range(10):
                grid_ids.append(f"{chr(ord('A') + xi)}{yi + 1}")
        return grid_ids

    def _grid_id(self, x_idx: int, y_idx: int) -> str:
        return f"{chr(ord('A') + x_idx)}{y_idx + 1}"

    def _grid_center(self, x_idx: int, y_idx: int) -> np.ndarray:
        x = (x_idx + 0.5) * self.grid_size
        y = (y_idx + 0.5) * self.grid_size
        return np.array([x, y], dtype=float)

    def _is_searching_state(self, uav: UAV) -> bool:
        # Grid painting should only happen in explicit scan behavior.
        return uav.current_skill in {"scan_sector"}

    def add_event(self, event_type: str, description: str, trigger_uav: str) -> None:
        self.trigger_events.append(
            {
                "event_type": event_type,
                "description": description,
                "trigger_uav": trigger_uav,
                "timestamp_sec": round(self.elapsed_time_sec, 2),
            }
        )

    @property
    def active_events(self) -> list[dict]:
        """Compatibility alias for event-driven loop checks."""
        return list(self.trigger_events)

    def _mark_grid_searched(self, grid_id: str) -> None:
        ratio = self.grid_coverage_ratio.get(grid_id, 0.0)
        if ratio >= self.search_complete_threshold:
            if grid_id in self.unexplored_grids:
                self.unexplored_grids.remove(grid_id)
            self.searched_grids.add(grid_id)
        else:
            self.searched_grids.discard(grid_id)
            self.unexplored_grids.add(grid_id)

    def _world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        px = int(np.clip(x / self.paint_resolution_m, 0, self.paint_width - 1))
        py = int(np.clip(y / self.paint_resolution_m, 0, self.paint_height - 1))
        return px, py

    def _paint_circle_at(
        self,
        x: float,
        y: float,
        radius: float,
        clip_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        # Slightly shrink brush radius to avoid raster edge appearing larger than analytic circle.
        effective_radius = max(0.0, radius - 0.6 * self.paint_resolution_m)
        if effective_radius <= 0.0:
            return
        cx, cy = self._world_to_pixel(x, y)
        r_px = max(1, int(math.ceil(effective_radius / self.paint_resolution_m)))

        x_min = max(0, cx - r_px)
        x_max = min(self.paint_width - 1, cx + r_px)
        y_min = max(0, cy - r_px)
        y_max = min(self.paint_height - 1, cy + r_px)

        if x_min > x_max or y_min > y_max:
            return

        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        xx, yy = np.meshgrid(xs, ys)
        dx = (xx - cx) * self.paint_resolution_m
        dy = (yy - cy) * self.paint_resolution_m
        inside = dx * dx + dy * dy <= effective_radius * effective_radius
        if clip_bounds is not None:
            x_min_b, x_max_b, y_min_b, y_max_b = clip_bounds
            xw = (xx.astype(float) + 0.5) * self.paint_resolution_m
            yw = (yy.astype(float) + 0.5) * self.paint_resolution_m
            inside &= (xw >= x_min_b) & (xw <= x_max_b) & (yw >= y_min_b) & (yw <= y_max_b)
        self.scan_paint_mask[y_min : y_max + 1, x_min : x_max + 1] |= inside

    def _paint_uav_path(self, uav: UAV) -> None:
        scan_alt = float(uav.skill_ctx.get("scan_altitude", uav.z))
        # Do not paint during climb/descend transition before entering scan altitude.
        if uav.z < scan_alt - 1.0:
            return

        prev = self._prev_positions.get(uav.uav_id, uav.position.copy())
        curr = uav.position.copy()
        radius = float(uav.fov_radius)
        seg_len = float(np.linalg.norm(curr[:2] - prev[:2]))
        step_len = max(2.0, radius * 0.35)
        steps = max(1, int(math.ceil(seg_len / step_len)))
        for i in range(steps + 1):
            t = i / steps
            p = prev + (curr - prev) * t
            self._paint_circle_at(float(p[0]), float(p[1]), radius, clip_bounds=None)

    def _recompute_grid_coverage_from_paint(self) -> None:
        for yi in range(self.rows):
            for xi in range(self.cols):
                gid = self._grid_id(xi, yi)
                x0 = int(xi * self.grid_size / self.paint_resolution_m)
                x1 = int((xi + 1) * self.grid_size / self.paint_resolution_m)
                y0 = int(yi * self.grid_size / self.paint_resolution_m)
                y1 = int((yi + 1) * self.grid_size / self.paint_resolution_m)
                x1 = min(x1, self.paint_width)
                y1 = min(y1, self.paint_height)
                cell_mask = self.scan_paint_mask[y0:y1, x0:x1]
                ratio = 0.0 if cell_mask.size == 0 else float(np.mean(cell_mask))
                self.grid_coverage_ratio[gid] = ratio
                self._mark_grid_searched(gid)

    def _update_grid_coverage(self, uav_list: list[UAV]) -> None:
        for uav in uav_list:
            if self._is_searching_state(uav):
                self._paint_uav_path(uav)
            self._prev_positions[uav.uav_id] = uav.position.copy()
        self._recompute_grid_coverage_from_paint()

    def _update_target_discovery(self, uav_list: list[UAV]) -> None:
        for uav in uav_list:
            for target in self.known_truth_targets:
                dist_xy = float(np.linalg.norm(target[:2] - uav.position[:2]))
                if dist_xy <= uav.fov_radius and self._is_searching_state(uav):
                    sigma = uav.sensor_noise
                    noisy_xy = target[:2] + np.random.normal(0.0, sigma, 2)
                    key = (round(float(target[0]), 1), round(float(target[1]), 1), round(float(target[2]), 1), uav.uav_id)
                    if key in self._known_target_keys:
                        continue

                    self._known_target_keys.add(key)
                    discovered = {
                        "detected_by": uav.uav_id,
                        "true_pos": [round(float(target[0]), 1), round(float(target[1]), 1), round(float(target[2]), 1)],
                        "measured_pos": [round(float(noisy_xy[0]), 1), round(float(noisy_xy[1]), 1), round(float(target[2]), 1)],
                        "sensor_noise": round(float(sigma), 2),
                    }
                    self.known_targets.append(discovered)
                    self._latest_target_estimate = {
                        "detected_by": discovered["detected_by"],
                        "measured_pos": discovered["measured_pos"],
                        "true_pos": discovered["true_pos"],
                        "timestamp_sec": round(self.elapsed_time_sec, 2),
                    }
                    self.add_event(
                        event_type="target_detected",
                        description=(
                            f"{uav.uav_id} detected target, measured="
                            f"({discovered['measured_pos'][0]}, {discovered['measured_pos'][1]}, {discovered['measured_pos'][2]})"
                        ),
                        trigger_uav=uav.uav_id,
                    )

    def _sync_descend_verify_targets(self, uav_list: list[UAV]) -> None:
        if not self._latest_target_estimate:
            return
        measured_pos = np.asarray(self._latest_target_estimate["measured_pos"], dtype=float).reshape(3)
        for uav in uav_list:
            if uav.current_skill == "descend_verify":
                uav.refresh_descend_verify_target(measured_pos)

    def _update_low_battery_events(self, uav_list: list[UAV]) -> None:
        for uav in uav_list:
            if uav.battery < 20.0 and uav.uav_id not in self._low_battery_uavs:
                self._low_battery_uavs.add(uav.uav_id)
                self.add_event(
                    event_type="low_battery",
                    description=f"{uav.uav_id} battery low: {uav.battery:.1f}%",
                    trigger_uav=uav.uav_id,
                )

    def update(self, uav_list: list[UAV], dt: float) -> None:
        self.elapsed_time_sec += float(dt)
        self._update_grid_coverage(uav_list)
        self._update_target_discovery(uav_list)
        self._sync_descend_verify_targets(uav_list)
        self._update_low_battery_events(uav_list)

    def render_ascii_grid(self) -> str:
        """
        Return grid text where searched='*' and unexplored='.'.
        Top row is y=10, bottom row is y=1.
        """
        lines: list[str] = []
        header = "    " + " ".join([chr(ord("A") + i) for i in range(10)])
        lines.append(header)
        for yi in range(9, -1, -1):
            row_cells: list[str] = []
            for xi in range(10):
                gid = self._grid_id(xi, yi)
                row_cells.append("*" if gid in self.searched_grids else ".")
            lines.append(f"{yi + 1:>2} | " + " ".join(row_cells))
        return "\n".join(lines)

    def _build_grid_coverage_matrix(self) -> list[list[float]]:
        matrix: list[list[float]] = []
        for yi in range(9, -1, -1):
            row: list[float] = []
            for xi in range(10):
                grid_id = self._grid_id(xi, yi)
                row.append(round(float(self.grid_coverage_ratio.get(grid_id, 0.0)), 2))
            matrix.append(row)
        return matrix

    def get_llm_state_json(self, uav_list: list[UAV]) -> str:
        active_events = list(self.trigger_events)
        self.trigger_events.clear()

        grid_coverage_matrix = self._build_grid_coverage_matrix()

        payload = {
            "mission_context": {
                "elapsed_time_sec": round(self.elapsed_time_sec, 1),
            },
            "world_memory": {
                "searched_grids": sorted(self.searched_grids),
                "unexplored_grids_count": len(self.unexplored_grids),
                "grid_coverage_matrix": grid_coverage_matrix,
                "grid_coverage_matrix_desc": "rows: y10->y1, cols: A->J, value: coverage ratio 0~1",
                "known_targets": self.known_targets,
                "latest_target_estimate": self._latest_target_estimate,
            },
            "fleet_telemetry": [
                {
                    "id": uav.uav_id,
                    "pos": [round(uav.x, 1), round(uav.y, 1), round(uav.z, 1)],
                    "current_skill": uav.current_skill,
                    "battery": round(uav.battery, 1),
                    "has_payload": uav.has_payload,
                    "has_grasped_target": uav.has_grasped_target,
                }
                for uav in uav_list
            ],
            "active_events": active_events,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
