from typing import Tuple


def compute_used_size(width: int, height: int, frame_size: int) -> Tuple[int, int]:
    if height < width:
        used_width = frame_size
        used_height = (frame_size * height) // width
    else:
        used_width = (frame_size * width) // height
        used_height = frame_size
    return used_width, used_height
