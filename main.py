"""
Entry point for Stage-1 static multi-UAV scene.
"""

import numpy as np

from llm_controller import LLMController
from sim.environment import EnvironmentConfig
from sim.global_state_manager import GlobalStateManager
from sim.uav import UAV
from viz.renderer import SceneRenderer3D


def build_initial_uavs() -> list[UAV]:
    """
    Build three UAV agents near origin with slight offsets
    to avoid visual overlap.
    """
    center_x = 500.0
    center_y = 500.0
    return [
        UAV(uav_id="UAV_A", position=np.array([center_x, center_y, 0.0])),
        UAV(uav_id="UAV_B", position=np.array([center_x + 6.0, center_y, 0.0])),
        UAV(uav_id="UAV_C", position=np.array([center_x, center_y + 6.0, 0.0])),
    ]


def main() -> None:
    env_cfg = EnvironmentConfig()
    uavs = build_initial_uavs()
    colors = ["red", "blue", "green"]
    discovery_target = np.array([865.0, 643.0, 0.0], dtype=float)
    draw_target_on_plot = True
    gsm = GlobalStateManager(
        map_width=1000.0,
        map_height=1000.0,
        grid_size=100.0,
        known_truth_targets=[discovery_target],
    )
    llm_controller = LLMController(
        config_path="llm_config.json",
        prompt_path="UAV_SKILLS.md",
    )

    renderer = SceneRenderer3D(
        env_cfg=env_cfg,
        arm_length_m=0.75,
        camera_fov_deg=84.0,
        max_uav_height_m=100.0,
        circle_samples=48,
        cone_theta_samples=20,
        cone_height_samples=10,
    )
    renderer.render_interactive_control(
        uavs=uavs,
        colors=colors,
        interval_ms=33,
        discovery_target=discovery_target,
        draw_target=draw_target_on_plot,
        global_state_manager=gsm,
        llm_json_every_n_frames=10,
        llm_json_output_path="llm_state_stream.jsonl",
        llm_controller=llm_controller,
        verbose_runtime_logs=False,
    )


if __name__ == "__main__":
    main()

