"""
UAV agent model (point-mass + inertia abstraction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


@dataclass
class UAV:
    """
    A simple UAV state model prepared for future closed-loop control.

    The vehicle is abstracted as a second-order inertial point mass with
    acceleration and speed limits.
    """

    uav_id: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    max_speed: float = 25.0  # m/s
    max_accel: float = 6.0  # m/s^2
    battery: float = 100.0
    has_payload: bool = False
    has_grasped_target: bool = False
    current_skill: str = "loiter"
    waypoints: list[np.ndarray] = field(default_factory=list)
    home_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    skill_ctx: dict[str, Any] = field(default_factory=dict)
    p_gain: float = 1.2
    min_flight_altitude: float = 10.0
    grasp_cruise_altitude: float = 50.0
    grasp_hold_seconds: float = 5.0
    grasp_capture_radius_m: float = 5.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        self.home_pos = np.asarray(self.home_pos, dtype=float).reshape(3)

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def z(self) -> float:
        return float(self.position[2])

    @property
    def vx(self) -> float:
        return float(self.velocity[0])

    @property
    def vy(self) -> float:
        return float(self.velocity[1])

    @property
    def vz(self) -> float:
        return float(self.velocity[2])

    def state_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Convenient packed state output."""
        return (self.x, self.y, self.z, self.vx, self.vy, self.vz)

    @property
    def fov_radius(self) -> float:
        return self.z * 1.5

    @property
    def sensor_noise(self) -> float:
        return self.z * 0.2

    def _consume_battery(self, moving: bool) -> None:
        if self.z <= 0.0:
            return
        if self.has_payload:
            cost = 0.03
        elif moving:
            cost = 0.01
        else:
            cost = 0.005
        self.battery = max(0.0, self.battery - cost)

    def _goto_step(self, target_pos: np.ndarray, dt: float) -> bool:
        target = np.asarray(target_pos, dtype=float).reshape(3)
        error = target - self.position
        dist = float(np.linalg.norm(error))
        if dist < 0.3:
            self.position = target.copy()
            self.velocity = np.zeros(3, dtype=float)
            return True

        desired_velocity = self.p_gain * error
        speed = float(np.linalg.norm(desired_velocity))
        if speed > self.max_speed:
            desired_velocity = desired_velocity / speed * self.max_speed

        accel_cmd = (desired_velocity - self.velocity) / dt
        accel_norm = float(np.linalg.norm(accel_cmd))
        if accel_norm > self.max_accel:
            accel_cmd = accel_cmd / accel_norm * self.max_accel

        self.velocity = self.velocity + accel_cmd * dt
        vel_norm = float(np.linalg.norm(self.velocity))
        if vel_norm > self.max_speed:
            self.velocity = self.velocity / vel_norm * self.max_speed

        step_vec = self.velocity * dt
        step_len = float(np.linalg.norm(step_vec))
        if step_len >= dist:
            self.position = target.copy()
            self.velocity = np.zeros(3, dtype=float)
            return True

        self.position = self.position + step_vec
        return False

    def _set_waypoint_skill(self, skill_name: str, waypoints: list[np.ndarray]) -> None:
        self.current_skill = skill_name
        self.waypoints = [np.asarray(wp, dtype=float).reshape(3) for wp in waypoints]

    def _build_transition_waypoints(
        self,
        target_xy: np.ndarray,
        target_z: float,
    ) -> list[np.ndarray]:
        """
        Build a smooth transition path:
        1) climb/hold to transit altitude,
        2) move horizontally at transit altitude,
        3) descend to target altitude if needed.
        """
        tx = float(target_xy[0])
        ty = float(target_xy[1])
        tz = float(target_z)
        dist_xy = float(np.linalg.norm(np.array([tx - self.x, ty - self.y], dtype=float)))
        # Any horizontal flight must happen above minimum flight altitude.
        if dist_xy > 1e-6:
            transit_z = max(self.z, tz, self.min_flight_altitude)
        else:
            transit_z = max(self.z, tz)

        waypoints: list[np.ndarray] = []
        if abs(transit_z - self.z) > 1e-6:
            waypoints.append(np.array([self.x, self.y, transit_z], dtype=float))

        if dist_xy > 1e-6:
            waypoints.append(np.array([tx, ty, transit_z], dtype=float))

        if abs(tz - transit_z) > 1e-6:
            waypoints.append(np.array([tx, ty, tz], dtype=float))

        if not waypoints:
            waypoints.append(np.array([tx, ty, tz], dtype=float))
        return waypoints

    def skill_loiter(self) -> None:
        """Emergency hover."""
        self.current_skill = "loiter"
        self.waypoints.clear()
        self.velocity = np.zeros(3, dtype=float)

    def skill_goto(self, target_pos: np.ndarray) -> None:
        """Fly in a straight line to a single target point."""
        target = np.asarray(target_pos, dtype=float).reshape(3)
        transition = self._build_transition_waypoints(target_xy=target[:2], target_z=float(target[2]))
        self._set_waypoint_skill("goto", transition)
        self.skill_ctx = {}

    def skill_scan_sector(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        altitude: float,
    ) -> None:
        """Generate lawnmower waypoints to cover a rectangular search area."""
        self.skill_scan_grids(
            grid_bounds=[(x_min, x_max, y_min, y_max)],
            altitude=altitude,
        )

    def _generate_lawnmower_points_rect(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        altitude: float,
    ) -> list[np.ndarray]:
        x0 = min(float(x_min), float(x_max))
        x1 = max(float(x_min), float(x_max))
        y0 = min(float(y_min), float(y_max))
        y1 = max(float(y_min), float(y_max))

        if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            return [np.array([cx, cy, altitude], dtype=float)]

        swath = max(6.0, 2.0 * (altitude * 1.5) * 0.85)
        y_values = np.arange(y0, y1 + swath, swath)

        lawn_points: list[np.ndarray] = []
        reverse = False
        for y in y_values:
            y_clamped = min(max(float(y), y0), y1)
            if reverse:
                lawn_points.append(np.array([x1, y_clamped, altitude], dtype=float))
                lawn_points.append(np.array([x0, y_clamped, altitude], dtype=float))
            else:
                lawn_points.append(np.array([x0, y_clamped, altitude], dtype=float))
                lawn_points.append(np.array([x1, y_clamped, altitude], dtype=float))
            reverse = not reverse

        # Remove consecutive duplicates caused by clamped y on last stripe.
        dedup: list[np.ndarray] = []
        for p in lawn_points:
            if not dedup or np.linalg.norm(p - dedup[-1]) > 1e-6:
                dedup.append(p)
        return dedup

    @staticmethod
    def _merge_scan_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        merged: list[tuple[float, float]] = []
        for x0, x1 in sorted((min(a, b), max(a, b)) for a, b in intervals):
            if not merged:
                merged.append((x0, x1))
                continue
            prev_x0, prev_x1 = merged[-1]
            if x0 <= prev_x1 + 1e-6:
                merged[-1] = (prev_x0, max(prev_x1, x1))
            else:
                merged.append((x0, x1))
        return merged

    def _build_union_scan_rows(
        self,
        grid_bounds: list[tuple[float, float, float, float]],
        altitude: float,
    ) -> tuple[list[tuple[float, list[tuple[float, float]]]], list[tuple[float, float, float, float]], float]:
        normalized_bounds: list[tuple[float, float, float, float]] = []
        for b in grid_bounds:
            x0 = min(float(b[0]), float(b[1]))
            x1 = max(float(b[0]), float(b[1]))
            y0 = min(float(b[2]), float(b[3]))
            y1 = max(float(b[2]), float(b[3]))
            normalized_bounds.append((x0, x1, y0, y1))

        if not normalized_bounds:
            return [], [], 0.0

        min_y = min(b[2] for b in normalized_bounds)
        max_y = max(b[3] for b in normalized_bounds)
        swath = max(6.0, 2.0 * (altitude * 1.5) * 0.85)
        y_values = np.arange(min_y, max_y + swath, swath)

        rows: list[tuple[float, list[tuple[float, float]]]] = []
        for y in y_values:
            y_clamped = min(max(float(y), min_y), max_y)
            row_segments = [
                (x0, x1)
                for x0, x1, y0, y1 in normalized_bounds
                if y0 - 1e-6 <= y_clamped <= y1 + 1e-6
            ]
            merged = self._merge_scan_intervals(row_segments)
            if merged:
                rows.append((y_clamped, merged))

        if not rows:
            center_y = 0.5 * (min_y + max_y)
            center_x = 0.5 * (min(b[0] for b in normalized_bounds) + max(b[1] for b in normalized_bounds))
            rows = [(center_y, [(center_x, center_x)])]

        return rows, normalized_bounds, swath

    def _build_scan_path_variant(
        self,
        rows: list[tuple[float, list[tuple[float, float]]]],
        altitude: float,
        start_from_top: bool,
        start_left: bool,
    ) -> list[np.ndarray]:
        ordered_rows = list(reversed(rows)) if start_from_top else list(rows)
        waypoints: list[np.ndarray] = []
        reverse_row = not start_left

        for y, intervals in ordered_rows:
            ordered_intervals = list(reversed(intervals)) if reverse_row else list(intervals)
            for x0, x1 in ordered_intervals:
                entry = np.array([x0, y, altitude], dtype=float) if not reverse_row else np.array([x1, y, altitude], dtype=float)
                exit_pt = np.array([x1, y, altitude], dtype=float) if not reverse_row else np.array([x0, y, altitude], dtype=float)
                if not waypoints or np.linalg.norm(entry - waypoints[-1]) > 1e-6:
                    waypoints.append(entry)
                if np.linalg.norm(exit_pt - waypoints[-1]) > 1e-6:
                    waypoints.append(exit_pt)
            reverse_row = not reverse_row

        dedup: list[np.ndarray] = []
        for p in waypoints:
            if not dedup or np.linalg.norm(p - dedup[-1]) > 1e-6:
                dedup.append(p)
        return dedup

    def skill_scan_grids(
        self,
        grid_bounds: list[tuple[float, float, float, float]],
        altitude: float,
    ) -> None:
        """Scan multiple grids as one unified search region."""
        altitude = max(0.0, float(altitude))
        if not grid_bounds:
            self.skill_loiter()
            return

        rows, normalized_bounds, _ = self._build_union_scan_rows(grid_bounds, altitude)
        variant_specs = [
            (False, True),
            (False, False),
            (True, True),
            (True, False),
        ]

        best_waypoints: list[np.ndarray] | None = None
        best_score = float("inf")
        cursor_xy = np.array([self.x, self.y], dtype=float)
        for start_from_top, start_left in variant_specs:
            candidate = self._build_scan_path_variant(rows, altitude, start_from_top, start_left)
            if not candidate:
                continue
            score = float(np.linalg.norm(candidate[0][:2] - cursor_xy))
            if len(candidate) > 1:
                score += 0.02 * float(np.sum(np.linalg.norm(np.diff(np.stack([p[:2] for p in candidate]), axis=0), axis=1)))
            if score < best_score:
                best_score = score
                best_waypoints = candidate

        if best_waypoints:
            entry = best_waypoints[0]
            transition = self._build_transition_waypoints(target_xy=entry[:2], target_z=altitude)
            full_waypoints = [*transition]
            for p in best_waypoints[1:]:
                if np.linalg.norm(p - full_waypoints[-1]) > 1e-6:
                    full_waypoints.append(p)
        else:
            b0 = normalized_bounds[0]
            center = np.array([0.5 * (b0[0] + b0[1]), 0.5 * (b0[2] + b0[3])], dtype=float)
            full_waypoints = self._build_transition_waypoints(target_xy=center, target_z=altitude)

        self._set_waypoint_skill("scan_sector", full_waypoints)
        self.skill_ctx = {"scan_altitude": altitude, "scan_grid_bounds": normalized_bounds}

    def skill_descend_verify(self, approx_pos: np.ndarray, verify_altitude: float) -> None:
        """Go above approximate location, then descend for close verification."""
        approx = np.asarray(approx_pos, dtype=float).reshape(3)
        verify_alt = max(0.0, float(verify_altitude))
        transition = self._build_transition_waypoints(target_xy=approx[:2], target_z=verify_alt)
        self._set_waypoint_skill("descend_verify", transition)
        self.skill_ctx = {
            "verify_target_xy": approx[:2].copy(),
            "verify_altitude": verify_alt,
        }

    def refresh_descend_verify_target(self, approx_pos: np.ndarray) -> None:
        """Update the active descend-verify target if a better estimate arrives."""
        if self.current_skill != "descend_verify":
            return
        approx = np.asarray(approx_pos, dtype=float).reshape(3)
        verify_alt = float(self.skill_ctx.get("verify_altitude", max(self.z, self.min_flight_altitude)))
        current_xy = np.asarray(self.skill_ctx.get("verify_target_xy", approx[:2]), dtype=float).reshape(2)
        if float(np.linalg.norm(current_xy - approx[:2])) < 1.0:
            return
        transition = self._build_transition_waypoints(target_xy=approx[:2], target_z=verify_alt)
        self._set_waypoint_skill("descend_verify", transition)
        self.skill_ctx.update({
            "verify_target_xy": approx[:2].copy(),
            "verify_altitude": verify_alt,
        })

    def skill_grasp_payload(self, exact_pos: np.ndarray) -> None:
        """Move to exact location, descend to ground, grasp, then return base."""
        exact = np.asarray(exact_pos, dtype=float).reshape(-1)
        if exact.shape[0] < 2:
            raise ValueError("exact_pos must contain at least x and y.")
        target_xy = np.array([float(exact[0]), float(exact[1])], dtype=float)

        # Fixed grasp profile: climb to 50m cruise -> fly -> descend to ground.
        cruise = max(self.grasp_cruise_altitude, self.min_flight_altitude)
        transition = self._build_transition_waypoints(target_xy=target_xy, target_z=cruise)
        transition.append(np.array([target_xy[0], target_xy[1], 0.0], dtype=float))
        self._set_waypoint_skill("grasp_payload", transition)
        self.skill_ctx = {
            "post_grasp_return_cruise_alt": max(cruise, self.z),
            "grasp_hold_remaining_sec": self.grasp_hold_seconds,
            "grasp_hold_done": False,
            "grasp_target_xy": target_xy.copy(),
            "grasp_capture_radius_m": self.grasp_capture_radius_m,
        }

    def skill_return_base(self, cruise_altitude: float) -> None:
        """Climb to safe altitude, then return to (0,0,0)."""
        cruise_alt = max(0.0, float(cruise_altitude))
        transit_alt = max(self.z, cruise_alt, self.min_flight_altitude)
        wp: list[np.ndarray] = []
        if abs(transit_alt - self.z) > 1e-6:
            wp.append(np.array([self.x, self.y, transit_alt], dtype=float))
        wp.append(np.array([self.home_pos[0], self.home_pos[1], transit_alt], dtype=float))
        if abs(self.home_pos[2] - transit_alt) > 1e-6:
            wp.append(np.array([self.home_pos[0], self.home_pos[1], self.home_pos[2]], dtype=float))
        self._set_waypoint_skill("return_base", wp)
        self.skill_ctx = {}

    def update_state(self, dt: float) -> None:
        """
        Main low-level execution loop ("cerebellum").
        State machine dispatches atomic skills and physics updates.
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if self.battery <= 0.0:
            self.skill_loiter()

        moved = False
        if self.current_skill == "loiter":
            self.velocity = np.zeros(3, dtype=float)
        elif self.current_skill == "grasp_payload":
            if self.waypoints:
                reached = self._goto_step(self.waypoints[0], dt)
                moved = True
                if reached:
                    self.waypoints.pop(0)
            else:
                self.velocity = np.zeros(3, dtype=float)
        elif self.waypoints:
            reached = self._goto_step(self.waypoints[0], dt)
            moved = True
            if reached:
                self.waypoints.pop(0)
        else:
            self.skill_loiter()

        if self.current_skill == "grasp_payload" and not self.waypoints:
            target_xy = np.asarray(self.skill_ctx.get("grasp_target_xy", self.position[:2]), dtype=float).reshape(2)
            capture_radius = float(self.skill_ctx.get("grasp_capture_radius_m", self.grasp_capture_radius_m))
            ground_dist = float(np.linalg.norm(self.position[:2] - target_xy))
            if ground_dist > capture_radius:
                # Still not close enough to capture: keep hovering and wait for the controller to reissue or adjust.
                self.skill_loiter()
                self._consume_battery(moving=False)
                return

            hold_done = bool(self.skill_ctx.get("grasp_hold_done", False))
            if not hold_done:
                remain = float(self.skill_ctx.get("grasp_hold_remaining_sec", self.grasp_hold_seconds))
                remain -= dt
                self.skill_ctx["grasp_hold_remaining_sec"] = max(0.0, remain)
                self.velocity = np.zeros(3, dtype=float)
                self.position[2] = 0.0
                if remain <= 0.0:
                    self.skill_ctx["grasp_hold_done"] = True
                else:
                    # Still holding on ground; do not transition yet.
                    self._consume_battery(moving=False)
                    return

            self.has_payload = True
            self.has_grasped_target = True
            print(f"[LLM STATE] {self.uav_id} grasped target successfully. Waiting for return-to-base command from LLM.")
            # Wait for LLM to decide whether to return base; do not auto-return.
            self.skill_loiter()

        if self.current_skill in {"goto", "scan_sector", "descend_verify", "return_base"} and not self.waypoints:
            self.skill_loiter()

        self.position[2] = max(0.0, self.position[2])
        self._consume_battery(moving=moved and np.linalg.norm(self.velocity) > 1e-6)

