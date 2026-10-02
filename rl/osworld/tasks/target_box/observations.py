from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .geometry import TargetBox, distance_to_box, point_in_box


def target_box_info(
    cursor_position: tuple[int, int],
    target_box: TargetBox,
) -> dict[str, Any]:
    return {
        "cursor_position": list(cursor_position),
        "target_box": target_box.as_dict(),
        "target_box_inside": point_in_box(cursor_position, target_box),
        "target_box_distance": round(distance_to_box(cursor_position, target_box), 3),
    }


def with_target_box_context(
    obs: Mapping[str, Any],
    target_box: TargetBox,
    cursor_position: tuple[int, int] | None,
) -> dict[str, Any]:
    enriched = dict(obs)
    enriched["target_box"] = target_box.as_dict()
    enriched["target_box_center"] = list(target_box_center(target_box))
    if cursor_position is not None:
        enriched["cursor_position"] = list(cursor_position)
    return enriched


def target_box_center(target_box: TargetBox) -> tuple[int, int]:
    return ((target_box.x1 + target_box.x2) // 2, (target_box.y1 + target_box.y2) // 2)


def cursor_position_from_obs(obs: Mapping[str, Any]) -> tuple[int, int] | None:
    value = obs.get("cursor_position")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = value
    if not isinstance(x, int) or not isinstance(y, int):
        return None
    return x, y
