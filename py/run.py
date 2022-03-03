from collections import deque
from oil import BackgroundSubtractor, DropTracker, FlowEstimator
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from typing import NamedTuple, Sequence
import argparse, gc, os
import cv2
import default_arguments, oil, utils
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_BACKGROUND_LEARNING_IN_SECONDS = 5
DEFAULT_SCALE_FACTOR = 0.5
DEFAULT_TRACKING_QUEUE_SIZE_IN_SECONDS = 2


BACKGROUND_ID = oil.BACKGROUND_ID
UNKNOWN_ID = -1
assert BACKGROUND_ID != UNKNOWN_ID


Device = torch.device


class BufferItem(NamedTuple):
    padded_img: Tensor
    flow: Tensor
    is_background: Tensor


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
def image_to_tensor(image: np.ndarray, device: Device) -> Tensor:
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
    with plt.style.context('default', after_reset=True):
        # Set style.
        BACKGROUND_ID_COLOR = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float32, device=args.device)
        DROP_ID_CMAP = torch.tensor(plt.cm.tab20.colors, dtype=torch.float32, device=args.device).T
        plt.rc('axes', titlesize=5, labelsize=2)
        plt.rc('xtick', labelsize=2)
        plt.rc('ytick', labelsize=2)
        plt.rc('lines', linewidth=0.1, markersize=0.5)
        plt.rc('grid', color='0.5', linestyle='-', linewidth=0.1)
        # Create the video capture object and open the input file.
        video_in = cv2.VideoCapture(args.input)
        try:
            frame_ind = 0
            num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video_in.get(cv2.CAP_PROP_FPS))
            width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
            used_width, used_height = utils.compute_used_size(width, height, args.frame_size)
            # Create and train the background subtractor.
            num_background_frames = args.background_learning * fps
            background_sample_lab = torch.zeros((num_background_frames, 3, used_height, used_width), dtype=torch.float32, device=args.device)
            with tqdm(desc=f'Filling the background queue (size = {args.background_learning} seconds)', total=num_background_frames) as pbar:
                ret, frame_bgr = video_in.read()
                while ret and frame_ind < num_background_frames:
                    frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                    frame_bgr /= 255
                    frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                    background_sample_lab[frame_ind, ...] = kornia.color.rgb_to_lab(frame_rgb)
                    frame_ind += 1
                    pbar.update(1)
                    ret, frame_bgr = video_in.read()
            if not ret:
                raise RuntimeError('End of video')
            print('Learning background... ', end='', flush=True)
            background_subtractor = BackgroundSubtractor(background_sample_lab)
            print('done.', flush=True)
            # Create the flow estimator.
            flow_estimator = FlowEstimator(args.device)
            tmp = max(32, int(32 / args.scale))
            padding = (0, ((used_width - 1) // tmp + 1) * tmp - used_width, 0, ((used_height - 1) // tmp + 1) * tmp - used_height)
            hsv = torch.zeros((3, used_height, used_width), dtype=torch.float32, device=args.device)
            hsv[1, ...] = 1.0
            # Create the drop tracker.
            num_tracking_queue_frames = int(args.tracking_queue_size * fps)
            drop_tracker = DropTracker(background_subtractor, patience=num_tracking_queue_frames)
            # Create the tracking queue and fill it.
            queue =  deque()
            with tqdm(desc=f'Filling the tracking queue (size = {args.tracking_queue_size} seconds)', total=num_tracking_queue_frames) as pbar:
                frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                padded_img = F.pad(frame_rgb.unsqueeze(0), padding)
                queue.append(BufferItem(padded_img=padded_img, flow=torch.zeros((used_height, used_width, 2), dtype=torch.float32, device=args.device), is_background=torch.full((used_height, used_width), True, dtype=torch.bool)))
                frame_ind += 1
                pbar.update(1)
                ret, frame_bgr = video_in.read()
                while ret and len(queue) < num_tracking_queue_frames:
                    # Compute flow and update the drop tracker.
                    frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                    frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                    frame_gray = kornia.color.bgr_to_grayscale(frame_bgr).squeeze(0)
                    padded_img = F.pad(frame_rgb.unsqueeze(0), padding)
                    flow, mask = flow_estimator.apply(queue[-1].padded_img, padded_img, args.scale)
                    mask = mask[0, 0, :used_height, :used_width]
                    flow = (flow[0, :2, :used_height, :used_width] * mask - flow[0, 2:, :used_height, :used_width] * (1 - mask)).permute(1, 2, 0)
                    drop_tracker.update(frame_rgb, frame_gray, frame_ind)
                    # Insert current frame in the frame queue.
                    queue.append(BufferItem(padded_img=padded_img, flow=flow, is_background=drop_tracker.is_background))
                    frame_ind += 1
                    pbar.update(1)
                    # Try to read the next frame.
                    ret, frame_bgr = video_in.read()
            if not ret:
                raise RuntimeError('End of video')
            # Create the video writer object.
            video_out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (used_width * 3, used_height))  #TODO Debug
            try:
                # Process the next frames and write the resulting video.
                area_counts = {}
                time_series = np.arange(num_frames, dtype=np.float32) / fps
                oil_spill = torch.full((used_height, used_width), BACKGROUND_ID, dtype=torch.int32)
                pixel_ind = torch.stack(torch.meshgrid(torch.arange(used_width, dtype=torch.long, device=args.device), torch.arange(used_height, dtype=torch.long, device=args.device), indexing='xy'), axis=2)
                with tqdm(desc=f'Main processing', total=(num_frames - (num_background_frames + num_tracking_queue_frames))) as pbar:
                    while ret:
                        # Compute flow and update the drop tracker.
                        frame_bgr = image_to_tensor(cv2.resize(frame_bgr, (used_width, used_height)), device=args.device)
                        frame_rgb = kornia.color.bgr_to_rgb(frame_bgr)
                        frame_gray = kornia.color.bgr_to_grayscale(frame_bgr).squeeze(0)
                        padded_img = F.pad(frame_rgb.unsqueeze(0), padding)
                        flow, mask = flow_estimator.apply(queue[-1].padded_img, padded_img, args.scale)
                        mask = mask[0, 0, :used_height, :used_width]
                        flow = (flow[0, :2, :used_height, :used_width] * mask - flow[0, 2:, :used_height, :used_width] * (1 - mask)).permute(1, 2, 0)
                        drop_tracker.update(frame_rgb, frame_gray, frame_ind)
                        ############################
                        ##### Look at the past #####
                        ############################
                        past_data: BufferItem = queue.popleft()
                        #TODO Debug
                        # # Draw lost drops
                        # if drop_tracker.lost_drops_footprint is not None:
                        #     is_lost_drop = drop_tracker.lost_drops_footprint != BACKGROUND_ID
                        #     oil_spill[is_lost_drop] = drop_tracker.lost_drops_footprint[is_lost_drop]
                        # # Follow the oil spill.
                        # spill_from = (past_data.flow + pixel_ind).to(torch.long)
                        # from_x = torch.clip(spill_from[..., 0].ravel(), 0, used_width - 1)
                        # from_y = torch.clip(spill_from[..., 1].ravel(), 0, used_height - 1)
                        # new_oil_spill = oil_spill[from_y, from_x].view(used_height, used_width)
                        # oil_spill = torch.maximum(oil_spill, new_oil_spill)
                        # oil_spill_rgb = BACKGROUND_ID_COLOR.view(3, 1, 1).repeat(1, used_height, used_width)
                        # is_oil = oil_spill > max(UNKNOWN_ID, BACKGROUND_ID)
                        # oil_spill_rgb[:, is_oil] = DROP_ID_CMAP[:, (oil_spill[is_oil].ravel() % len(DROP_ID_CMAP)).long()]
                        # # Update visible background statistics
                        # drop_ids, spill_areas = torch.unique(oil_spill[is_oil], return_counts=True)
                        # for drop_id, spill_area in zip(drop_ids, spill_areas):
                        #     area_series = area_counts.get(drop_id, None)
                        #     if area_series is None:
                        #         area_series = np.zeros((num_frames,), dtype=np.float32, device=args.device)
                        #         area_counts[drop_id] = area_series
                        #     area_series[frame_ind] = spill_area * (100 / (used_width * used_height))
                        # # Draw active drops.
                        # rectangle = []
                        # color = []
                        # for drop in drop_tracker.drops:
                        #     if drop['active']:
                        #         x, y, w, h = map(int, drop['bbox'])
                        #         rectangle.append((x, y, x + w, y + h))
                        #         color.append((1.0, 0.0, 1.0) if drop['last_seen'] == frame_ind else (0.5, 0.5, 0.5))
                        # if len(rectangle) > 0:
                        #     kornia.utils.draw_rectangle(frame_rgb, rectangle=torch.as_tensor(rectangle, dtype=torch.float32, device=args.device), color=torch.as_tensor(color, dtype=torch.float32, device=args.device))
                        # # Plot data.
                        # fig, (ax_area, ax_drop) = plt.subplots(2, 1, dpi=300, figsize=(2 * used_width / 300, used_height / 300))
                        # try:
                        #     # Area series.
                        #     if len(area_counts) > 0:
                        #         series = []
                        #         colors = []
                        #         for drop_id, area_series in area_counts.items():
                        #             series.append(area_series)
                        #             color = DROP_ID_CMAP[:, drop_id % len(DROP_ID_CMAP)]
                        #             colors.append(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
                        #         ax_area.stackplot(time_series, *series, colors=colors)
                        #     ax_area.set_title('Areas')
                        #     ax_area.axes.set_xlabel('Time (seconds)')
                        #     ax_area.axes.set_ylabel('Area (%)')
                        #     ax_area.axes.set_ylim(0.0, 105.0)
                        #     ax_area.grid(True)
                        #     # Drop.
                        #     max_id = 0
                        #     for drop in drop_tracker.drops:
                        #         id = drop['id']
                        #         plt.plot((time_series[drop['first_seen']], time_series[drop['last_seen']]), (id, id), marker='x', color=DROP_ID_CMAP[:, id % len(DROP_ID_CMAP)])
                        #         max_id = max(max_id, id)
                        #     ax_drop.set_title('Detected Drops')
                        #     ax_drop.axes.set_xlabel('Time (seconds)')
                        #     ax_drop.axes.set_ylabel('Drop')
                        #     ax_drop.axes.set_ylim(0, max_id + 1)
                        #     ax_drop.axes.set_yticks(np.arange(max_id + 1))
                        #     ax_drop.axes.set_yticklabels(['', *map(str, range(1, max_id + 1))])
                        #     ax_drop.grid(True)
                        #     # Draw plots.
                        #     plt.tight_layout()
                        #     fig.canvas.draw()
                        #     analysis_rgb = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
                        # finally:
                        #     plt.close(fig)
                        #     gc.collect()
                        # Write result to output video.
                        video_out.write(np.concatenate((
                            tensor_to_image(kornia.color.rgb_to_bgr(past_data.padded_img[0, :, :used_height, :used_width])),
                            tensor_to_image(kornia.color.rgb_to_bgr(flow_to_rgb(past_data.flow, hsv))),
                            tensor_to_image(past_data.is_background.to(torch.float32).unsqueeze(0).expand(3, -1, -1)),  #TODO Debug
                            #TODO Debug
                            # tensor_to_image(kornia.color.rgb_to_bgr(oil_spill_rgb)),
                            # analysis_rgb[:, :, ::-1],
                        ), axis=1))
                        ###############################
                        ##### Back to the present #####
                        ###############################
                        # Update the frame queue.
                        queue.append(BufferItem(padded_img=padded_img, flow=flow, is_background=drop_tracker.is_background))
                        frame_ind += 1
                        pbar.update(1)
                        # Try to read the next frame.
                        ret, frame_bgr = video_in.read()
            finally:
                video_out.release()
        finally:
            video_in.release()


if __name__ == '__main__':
    # Define helper functions for parsing arguments.
    def positiv_float(arg: str) -> float:
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
    # Declare background subtraction arguments.
    group = parser.add_argument_group('background subtraction arguments')
    group.add_argument('--background_learning', metavar='TIME', type=default_arguments.positive_int_type, default=DEFAULT_BACKGROUND_LEARNING_IN_SECONDS, help='time (in seconds) at the beginning of the video for background learning')
    # Declare flow estimation arguments.
    group = parser.add_argument_group('flow estimation arguments')
    group.add_argument('--scale', metavar='FACTOR', type=positiv_float, default=DEFAULT_SCALE_FACTOR, help='scale factor for flow estimtion')
    # Declare drop detection/tracking arguments.
    group = parser.add_argument_group('drop detection/tracking arguments')
    group.add_argument('--tracking_queue_size', metavar='TIME', type=default_arguments.positive_int_type, default=DEFAULT_TRACKING_QUEUE_SIZE_IN_SECONDS, help='time (in seconds) before stop tracking the drop')
    # Declare other arguments.
    group = parser.add_argument_group('other arguments')
    group.add_argument(*default_arguments.DEVICE_ARGS, **default_arguments.DEVICE_KWARGS)
    # Parse arguments.
    args = parser.parse_args(['--input', os.path.join('..', 'FeicoesOleosas-Dataset', 'Petrobras - P51', 'P51-8S.mp40')])  #TODO Debug
    if args.output is None:
        args.output = f'{os.path.splitext(args.input)[0]}-output.mp4'
    # Call the main method.
    main(args)
