from __future__ import annotations

import io
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TargetBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class TargetBoxConfig:
    box_width: int = 150
    box_height: int = 150
    margin: int = 40
    cursor_margin: int = 20
    seed: int = 0


def sample_target_box(
    config: TargetBoxConfig,
    *,
    screen_width: int,
    screen_height: int,
    instance_key: str,
) -> TargetBox:
    _validate_config(config, screen_width=screen_width, screen_height=screen_height)
    rng = random.Random(f"{config.seed}:{instance_key}:box")
    x1 = rng.randint(config.margin, screen_width - config.margin - config.box_width)
    y1 = rng.randint(config.margin, screen_height - config.margin - config.box_height)
    return TargetBox(
        x1=x1,
        y1=y1,
        x2=x1 + config.box_width - 1,
        y2=y1 + config.box_height - 1,
    )


def sample_cursor_start(
    config: TargetBoxConfig,
    box: TargetBox,
    *,
    screen_width: int,
    screen_height: int,
    instance_key: str,
) -> tuple[int, int]:
    _validate_config(config, screen_width=screen_width, screen_height=screen_height)
    rng = random.Random(f"{config.seed}:{instance_key}:cursor")
    min_x = config.cursor_margin
    max_x = screen_width - config.cursor_margin - 1
    min_y = config.cursor_margin
    max_y = screen_height - config.cursor_margin - 1

    for _ in range(100):
        point = (rng.randint(min_x, max_x), rng.randint(min_y, max_y))
        if not point_in_box(point, box):
            return point

    candidates = [
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
    ]
    outside = [point for point in candidates if not point_in_box(point, box)]
    if not outside:
        raise ValueError("target_box leaves no valid cursor start outside the box")
    return max(outside, key=lambda point: distance_to_box(point, box))


def point_in_box(point: tuple[int, int], box: TargetBox) -> bool:
    x, y = point
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


def distance_to_box(point: tuple[int, int], box: TargetBox) -> float:
    x, y = point
    dx = max(box.x1 - x, 0, x - box.x2)
    dy = max(box.y1 - y, 0, y - box.y2)
    return math.hypot(dx, dy)


def annotate_observation(
    obs: Mapping[str, Any],
    box: TargetBox,
) -> dict[str, Any]:
    annotated = dict(obs)
    annotated["screenshot"] = annotate_screenshot_bytes(_screenshot_bytes(obs), box)
    return annotated


def annotate_screenshot_bytes(screenshot: bytes, box: TargetBox) -> bytes:
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = (0, 255, 0)
    draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=3)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def screenshot_size(obs: Mapping[str, Any]) -> tuple[int, int]:
    with Image.open(io.BytesIO(_screenshot_bytes(obs))) as image:
        return image.size


def _screenshot_bytes(obs: Mapping[str, Any]) -> bytes:
    screenshot = obs.get("screenshot")
    if not isinstance(screenshot, bytes):
        raise ValueError("OSWorld observation has no screenshot bytes")
    return screenshot


def _validate_config(
    config: TargetBoxConfig,
    *,
    screen_width: int,
    screen_height: int,
) -> None:
    if config.box_width <= 0 or config.box_height <= 0:
        raise ValueError("target_box dimensions must be positive")
    if config.margin < 0 or config.cursor_margin < 0:
        raise ValueError("target_box margins must be non-negative")
    if config.box_width + 2 * config.margin > screen_width:
        raise ValueError("target_box width and margin do not fit screen")
    if config.box_height + 2 * config.margin > screen_height:
        raise ValueError("target_box height and margin do not fit screen")
    if 2 * config.cursor_margin >= screen_width:
        raise ValueError("target_box cursor_margin does not fit screen width")
    if 2 * config.cursor_margin >= screen_height:
        raise ValueError("target_box cursor_margin does not fit screen height")
