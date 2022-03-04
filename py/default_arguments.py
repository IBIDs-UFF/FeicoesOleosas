from typing import Tuple
import argparse, os, warnings
import torch


def compute_used_size(width: int, height: int, frame_size: int) -> Tuple[int, int]:
    if height < width:
        used_width = frame_size
        used_height = (frame_size * height) // width
    else:
        used_width = (frame_size * width) // height
        used_height = frame_size
    return used_width, used_height


def dirpath_type(arg: str) -> str:
    if not os.path.isdir(arg):
        raise argparse.ArgumentTypeError(f'folder "{arg}" not found')
    return arg


def device_type(arg: str) -> torch.device:
    try:
        if arg.lower() == 'cpu':
            warnings.warn('CPU processing is very slow. Choose "--device cuda" for better performance.', category=RuntimeWarning)
        return torch.device(arg)
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))


def filepath_type(arg: str) -> str:
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError(f'file "{arg}" not found')
    return arg


def positive_int_type(arg: str) -> int:
    try:
        arg = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{arg} not an integer literal')
    if arg <= 0:
        raise argparse.ArgumentTypeError(f'{arg} not positive')
    return arg


DATASET_ARGS = ['--dataset']
DATASET_KWARGS = dict(metavar='PATH', type=dirpath_type, required=True, help='path to the dataset folder')

DEVICE_ARGS = ['--device']
DEVICE_KWARGS = dict(metavar='NAME', type=device_type, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='device used for processing')

FRAME_SIZE_ARGS = ['--frame_size']
FRAME_SIZE_KWARGS = dict(metavar='SIZE', type=positive_int_type, default=640, help='the longer side of the frame is resized to SIZE while maintaining the aspect ratio')

INPUT_ARGS = ['--input']
INPUT_KWARGS = dict(metavar='PATH', type=filepath_type, required=True, help='path to the input .mp4 file')
