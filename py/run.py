import os, sys
YOLO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'oil', 'yolov5'))
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)

from collections import deque
from oil import DropTracker, FlowEstimator
from torch import Tensor
from tqdm import tqdm
from typing import NamedTuple
import argparse
import cv2
import default_arguments, oil
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch


DEFAULT_CONFIDENCE_THRESHOLD = 0.10
DEFAULT_LIFETIME_THRESHOLD_IN_SECONDS = 4.0
DEFAULT_SCALE_FACTOR = 4.0
DEFAULT_TRACKING_QUEUE_SIZE_IN_SECONDS = 1.0


class BufferItem(NamedTuple):
    frame_rgb: Tensor
    padded_img: Tensor
    flow: Tensor


def default_output_mp4_path(input: str) -> str:
    return f'{os.path.splitext(input)[0]}-output.mp4'


def default_output_xlsx_path(output: str) -> str:
    return f'{os.path.splitext(output)[0]}.xlsx'


@torch.no_grad()
def flow_to_rgb(flow: Tensor, hsv: Tensor) -> Tensor:
    # Hue channel encodes angle in [0, 2*pi] range.
    ang = torch.atan2(flow[..., 1], flow[..., 0])
    ang[ang < 0.0] += 2 * torch.pi
    hsv[0, ...] = ang
    # Value channel encodes magnitude in [mag_min, mag_max] -> [0, 1] range.
    mag = torch.linalg.vector_norm(flow, dim=2)
    mag_min, mag_max = mag.min(), mag.max()
    hsv[2, ...] = (mag - mag_min) / (mag_max - mag_min)
    # Convert from HSV to RGB.
    return kornia.color.hsv_to_rgb(hsv)


@torch.no_grad()
def image_to_tensor(image: np.ndarray, device: torch.device) -> Tensor:
    if image.ndim == 2:
        image = image[:, :, None]
    image = torch.from_numpy(image.transpose(2, 0, 1)).contiguous()
    if isinstance(image, torch.ByteTensor):
        return image.to(dtype=torch.float32, device=device) / 255
    else:
        return image.to(device=device)


@torch.no_grad()
def tensor_to_image(tensor: Tensor) -> np.ndarray:
    if tensor.ndim != 2:
        tensor = tensor.permute(1, 2, 0)
    if isinstance(tensor, torch.ByteTensor):
        return tensor.cpu().numpy()
    else:
        return (tensor * 255).to(dtype=torch.uint8).cpu().numpy()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    # Create the video capture object and open the input file.
    video_in = cv2.VideoCapture(args.input)
    try:
        frame_ind = 0
        num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video_in.get(cv2.CAP_PROP_FPS))
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        used_width, used_height = default_arguments.compute_used_size(width, height, args.frame_size)
        # Set style.
        BACKGROUND_ID_COLOR = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float32, device=args.device)
        DROP_ID_CMAP = torch.tensor(plt.cm.tab20.colors, dtype=torch.float32, device=args.device).T
        frame_title_style = dict(org=(10, used_height - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, lineType=cv2.LINE_AA)
        # Create the flow estimator.
        flow_estimator = FlowEstimator(device=args.device, scale=args.scale, used_width=used_width, used_height=used_height)
        tmp = max(32, int(32 / args.scale))
        padding = (0, ((used_width - 1) // tmp + 1) * tmp - used_width, 0, ((used_height - 1) // tmp + 1) * tmp - used_height)
        delta = torch.zeros((used_height, used_width, 2), dtype=torch.float32, device=args.device)
        hsv = torch.zeros((3, used_height, used_width), dtype=torch.float32, device=args.device)
        hsv[1, ...] = 1.0
        # Create the drop tracker.
        num_tracking_queue_frames = int(args.tracking_queue_size * fps)
        num_lifetime_threshold_frames = int(args.lifetime_threshold * fps)
        drop_tracker = DropTracker(confidence_threshold=args.confidence_threshold, device=args.device, frame_size=args.frame_size, lifetime_threshold=num_lifetime_threshold_frames, patience=num_tracking_queue_frames)
        # Create the tracking queue and fill it.
        queue = deque()
        with tqdm(desc=f'Filling the tracking queue with {args.tracking_queue_size} second(s)', total=num_tracking_queue_frames) as pbar:
            ret, frame_bgr = video_in.read()
            if ret:
                frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                padded_img = torch.nn.functional.pad(frame_rgb.unsqueeze(0), padding)
                queue.append(BufferItem(frame_rgb=frame_rgb, padded_img=padded_img, flow=torch.zeros((used_height, used_width, 2), dtype=torch.float32, device=args.device)))
                frame_ind += 1
                pbar.update(1)
                # Try to read the next frame.
                ret, frame_bgr = video_in.read()
            while ret and len(queue) < num_tracking_queue_frames:
                # Compute flow and update the drop tracker.
                frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                padded_img = torch.nn.functional.pad(frame_rgb.unsqueeze(0), padding)
                flow = flow_estimator.estimate(queue[-1].padded_img, padded_img)
                drop_tracker.update(frame_rgb, frame_ind)
                # Insert current frame in the frame queue.
                queue.append(BufferItem(frame_rgb=frame_rgb, padded_img=padded_img, flow=flow))
                frame_ind += 1
                pbar.update(1)
                # Try to read the next frame.
                ret, frame_bgr = video_in.read()
        if not ret:
            raise RuntimeError('End of video')
        # Create the video writer object.
        video_out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (used_width * 3, used_height))
        try:
            # Process the next frames and write the resulting video.
            area_counts = {}
            oil_spill = torch.full((used_height, used_width), oil.BACKGROUND_ID, dtype=torch.int32, device=args.device)
            pixel_ind = torch.stack(torch.meshgrid(torch.arange(used_width, dtype=torch.long, device=args.device), torch.arange(used_height, dtype=torch.long, device=args.device), indexing='xy'), axis=2)
            with tqdm(desc=f'Main processing', total=(num_frames - num_tracking_queue_frames)) as pbar:
                while ret:
                    # Compute flow and update the drop tracker.
                    frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                    frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                    padded_img = torch.nn.functional.pad(frame_rgb.unsqueeze(0), padding)
                    flow = flow_estimator.estimate(queue[-1].padded_img, padded_img)
                    drop_tracker.update(frame_rgb, frame_ind)
                    # Draw lost drops
                    if drop_tracker.lost_drops_footprint is not None:
                        is_lost_drop = drop_tracker.lost_drops_footprint != oil.BACKGROUND_ID
                        oil_spill[is_lost_drop] = drop_tracker.lost_drops_footprint[is_lost_drop]
                    # Follow the oil spill.
                    past_data: BufferItem = queue.popleft()
                    delta += past_data.flow
                    delta_round = delta.round()
                    spill_from = (delta_round + pixel_ind).to(torch.long)
                    from_x = torch.clip(spill_from[..., 0].ravel(), 0, used_width - 1)
                    from_y = torch.clip(spill_from[..., 1].ravel(), 0, used_height - 1)
                    oil_spill = torch.maximum(oil_spill[from_y, from_x].view(used_height, used_width), oil_spill)
                    delta[delta_round != 0.0] = 0.0
                    # Update visible background statistics
                    oil_spill_id = oil_spill.ravel()
                    drop_ids, spill_areas = torch.unique(oil_spill_id[oil_spill_id != oil.BACKGROUND_ID], return_counts=True)
                    for drop_id, spill_area in zip(drop_ids, spill_areas):
                        drop_id, spill_area = drop_id.item(), spill_area.item()
                        area_series = area_counts.get(drop_id, None)
                        if area_series is None:
                            area_series = np.zeros((num_frames,), dtype=np.float32)
                            area_counts[drop_id] = area_series
                        area_series[frame_ind] = spill_area * (100 / (used_width * used_height))
                    # Draw drop bounding boxes.
                    rectangle = []
                    color = []
                    drops_count = 0
                    candidate_drops_count = 0
                    for drop in drop_tracker.drops:
                        if drop['id'] is None:
                            x, y, w, h = drop['bbox']
                            rectangle.append((x, y, x + w, y + h))
                            color.append((1.0, 0.0, 1.0) if drop['last_seen'] == frame_ind else (0.5, 0.5, 0.5))
                            candidate_drops_count += 1
                        else:
                            drops_count += 1
                    if len(rectangle) > 0:
                        kornia.utils.draw_rectangle(frame_rgb.unsqueeze(0), rectangle=torch.as_tensor([rectangle], dtype=torch.float32, device=args.device), color=torch.as_tensor([color], dtype=torch.float32, device=args.device))
                    # Draw oil spill.
                    oil_spill_rgb = DROP_ID_CMAP[:, (oil_spill_id % DROP_ID_CMAP.shape[-1]).long()]
                    oil_spill_rgb[:, oil_spill_id == oil.BACKGROUND_ID] = BACKGROUND_ID_COLOR.unsqueeze(1)
                    oil_spill_rgb = oil_spill_rgb.view(3, used_height, used_width)
                    # Write result to output video.
                    video_out.write(np.concatenate((
                        cv2.putText(tensor_to_image(kornia.color.rgb_to_bgr(past_data.frame_rgb)).copy(), f'Drops: {drops_count} | Candidates: {candidate_drops_count}', color=(0, 0, 0), **frame_title_style),
                        cv2.putText(tensor_to_image(kornia.color.rgb_to_bgr(flow_to_rgb(past_data.flow, hsv))).copy(), 'Motion Flow', color=(255, 255, 255), **frame_title_style),
                        cv2.putText(tensor_to_image(kornia.color.rgb_to_bgr(oil_spill_rgb)).copy(), 'Oil Spill', color=(0, 0, 0), **frame_title_style),
                    ), axis=1))
                    # Update the frame queue.
                    queue.append(BufferItem(frame_rgb=frame_rgb, padded_img=padded_img, flow=flow))
                    frame_ind += 1
                    pbar.update(1)
                    # Try to read the next frame.
                    ret, frame_bgr = video_in.read()
        finally:
            video_out.release()
            # Write result to .xlsx file.
            time = list(range(1, num_frames // fps))
            drops = []
            for drop in drop_tracker.drops:
                drop_id = drop['id']
                if drop_id is not None:
                    drop['first_seen'] /= fps
                    drop['last_seen'] /= fps
                    drop['lifetime'] /= fps
                    area_series = area_counts[drop_id]
                    area = [area_series[ind:(ind + fps)].mean().item() for ind in range(0, num_frames, fps)]
                    for key, value in zip(time, area):
                        drop[str(key)] = value
                    drops.append(drop)
            df = pandas.DataFrame(drops, columns=['id', 'first_seen', 'last_seen', 'lifetime', *map(str, time)])
            df.to_excel(default_output_xlsx_path(args.output), sheet_name='Drops', float_format='%.2f', index=False)
    finally:
        video_in.release()


if __name__ == '__main__':
    # Define helper functions for parsing arguments.
    def float_in_range_type(arg: str, min: float, max: float) -> float:
        try:
            arg = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{arg} not a floating-point literal')
        if arg < min or max < arg:
            raise argparse.ArgumentTypeError(f'{arg} not in range [{min}, {max}]')
        return arg
    def positive_float_type(arg: str) -> float:
        try:
            arg = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{arg} not a floating-point literal')
        if arg <= 0.0:
            raise argparse.ArgumentTypeError(f'{arg} not positive')
        return arg
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    # Declare input/output arguments.
    group = parser.add_argument_group('input/output arguments')
    group.add_argument(*default_arguments.INPUT_ARGS, **default_arguments.INPUT_KWARGS)
    group.add_argument('--output', metavar='PATH', type=str, help='path to the resulting .mp4 file')
    group.add_argument(*default_arguments.FRAME_SIZE_ARGS, **default_arguments.FRAME_SIZE_KWARGS)
    # Declare flow estimation arguments.
    group = parser.add_argument_group('flow estimation arguments')
    group.add_argument('--scale', metavar='FACTOR', type=positive_float_type, default=DEFAULT_SCALE_FACTOR, help='scale factor for flow estimtion')
    # Declare drop detection/tracking arguments.
    group = parser.add_argument_group('drop detection and tracking arguments')
    group.add_argument('--confidence_threshold', metavar='VALUE', type=lambda arg: float_in_range_type(arg, 0.0, 1.0), default=DEFAULT_CONFIDENCE_THRESHOLD, help='confidence threshold')
    group.add_argument('--lifetime_threshold', metavar='TIME', type=positive_float_type, default=DEFAULT_LIFETIME_THRESHOLD_IN_SECONDS, help='minimum lifetime (in seconds) for a drop candidate be considered a drop')
    group.add_argument('--tracking_queue_size', metavar='TIME', type=positive_float_type, default=DEFAULT_TRACKING_QUEUE_SIZE_IN_SECONDS, help='time (in seconds) before stop tracking the drop')
    # Declare other arguments.
    group = parser.add_argument_group('other arguments')
    group.add_argument(*default_arguments.DEVICE_ARGS, **default_arguments.DEVICE_KWARGS)
    # Parse arguments.
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output_mp4_path(args.input)
    # Call the main method.
    main(args)
