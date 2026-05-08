"""
Environment definitions for the UAV simulation stage-1 scene.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """3D world boundaries and camera setup."""

    x_min: float = 0.0
    x_max: float = 1000.0
    y_min: float = 0.0
    y_max: float = 1000.0
    z_min: float = 0.0
    z_max: float = 100.0

    # A comfortable oblique view: slight top-down angle.
    elev_deg: float = 28.0
    azim_deg: float = -58.0

