from oil import DropTracker
from tqdm import tqdm
import cv2
import gc, glob, os
import matplotlib.pyplot as plt
import numpy as np
import oil

#TODO Manter buffer de frames por causa da patience


DATASET_DIR = os.path.join('..', 'FeicoesOleosas-Dataset', 'Petrobras')

DEFAULT_PATIENCE_IN_SECONDS = 2
BACKGROUND_LEARNING_IN_SECONDS = 30
SKIP_SECONDS = 0

BACKGROUND_ID = 0
UNKNOWN_ID = -1

BACKGROUND_ID_COLOR = np.asarray((255, 255, 255), dtype=np.uint8)
UNKNOWN_ID_COLOR = np.asarray((0, 0, 0), dtype=np.uint8)
DROP_ID_CMAP = (np.asarray(plt.cm.tab20.colors) * 255).astype(np.uint8)


def main() -> None:
    scale_factor = 0.25
    # Set style.
    plt.rc('axes', titlesize=5, labelsize=2)
    plt.rc('xtick', labelsize=2)
    plt.rc('ytick', labelsize=2)
    plt.rc('lines', linewidth=0.1, markersize=0.5)
    plt.rc('grid', color='0.5', linestyle='-', linewidth=0.1)
    # For each input video...
    for video_path in sorted(glob.glob(os.path.join(DATASET_DIR, '*.mp4'))):
        basename = os.path.basename(video_path)
        # Create the video capture object and open the input file.
        with oil.VideoReader(video_path) as video:
            pbar = iter(tqdm(video, desc=f'Processing "{basename}"', total=video.frame_count))
            used_width = int(video.width * scale_factor)
            used_height = int(video.height * scale_factor)
            # Create the video writer object and open the output file.
            with oil.VideoWriter(f'{os.path.splitext(basename)[0]}-flow.mp4', (4 * used_width, used_height), video.fps) as out:
                # Create the background subtractor.
                background_subtractor = cv2.createBackgroundSubtractorMOG2()
                num_background_samples = BACKGROUND_LEARNING_IN_SECONDS * video.fps
                for _ in range(num_background_samples):
                    background_subtractor.apply(cv2.resize(next(pbar), (used_width, used_height)))
                # Create a drop tracker.
                tracker = DropTracker(background_subtractor, patience=int(DEFAULT_PATIENCE_IN_SECONDS * video.fps))
                # Skip the first part of boring videos (for debug only).
                num_boring_frames = SKIP_SECONDS * video.fps
                for _ in range(max(num_boring_frames - num_background_samples, 0)):
                    next(pbar)
                # Initialize some useful variables.
                frame_flow = None
                pingpong = 0
                time_series = np.arange(video.frame_count, dtype=np.float32) / video.fps
                visible_backbround_series = np.zeros((video.frame_count,), dtype=np.float32)
                pixel_ind = np.stack(np.meshgrid(np.arange(used_width, dtype=np.int32), np.arange(used_height, dtype=np.int32), indexing='xy'), axis=2)
                # Render output video.
                previous_frame_rgb = cv2.resize(next(pbar), (used_width, used_height))
                previous_frame_gray = cv2.cvtColor(previous_frame_rgb, cv2.COLOR_RGB2GRAY)
                previous_oil_spill = np.full((used_height, used_width), BACKGROUND_ID, dtype=np.int32)
                oil_spill = np.full((used_height, used_width), BACKGROUND_ID, dtype=np.int32)
                for frame_ind in range(max(num_boring_frames, num_background_samples) + 1, video.frame_count):
                    frame_rgb = cv2.resize(next(pbar), (used_width, used_height))
                    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    # Find and track drops.
                    tracker.update(frame_rgb, frame_gray, frame_ind)
                    ## frame_segmentation_rgb = np.stack((tracker.foreground_msk, tracker.drop_msk, tracker.shadow_msk), axis=2)
                    for drop in tracker.drops:
                        if drop['active']:
                            x, y, w, h = map(int, drop['bbox'])
                            color = (255, 0, 255) if drop['last_seen'] == frame_ind else (127, 127, 127)
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 1)
                    # Follow the oil spill.
                    frame_flow = cv2.calcOpticalFlowFarneback(frame_gray, previous_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    spill_from = (frame_flow + pixel_ind + 0.5).astype(np.int32)
                    is_unknown_id = np.logical_or(np.logical_or(np.logical_or(spill_from[..., 0] <= 0, spill_from[..., 0] >= (used_width - 1)), spill_from[..., 1] <= 0), spill_from[..., 1] >= (used_height - 1))
                    is_not_unknown_id = np.logical_not(is_unknown_id)
                    oil_spill[is_unknown_id] = UNKNOWN_ID
                    copied_from = spill_from[is_not_unknown_id, :]
                    oil_spill[is_not_unknown_id] = previous_oil_spill[copied_from[:, 1], copied_from[:, 0]]
                    if tracker.lost_drops_footprint is not None:
                        is_lost_drop = tracker.lost_drops_footprint != BACKGROUND_ID
                        oil_spill[is_lost_drop] = tracker.lost_drops_footprint[is_lost_drop]
                    is_drop_id = oil_spill > max(BACKGROUND_ID, UNKNOWN_ID)
                    is_background_id = np.logical_or(oil_spill == BACKGROUND_ID, np.logical_and(is_unknown_id, tracker.background_msk))
                    oil_spill_rgb = np.full((used_height, used_width, 3), UNKNOWN_ID_COLOR, dtype=np.uint8)
                    oil_spill_rgb[is_background_id, :] = BACKGROUND_ID_COLOR
                    #TODO
                    # oil_spill_rgb[is_drop_id, :] = DROP_ID_CMAP[oil_spill[is_drop_id] % len(DROP_ID_CMAP), :]
                    if is_drop_id.any():
                        oil_spill_rgb[..., :] = np.stack(((255 * is_drop_id).astype(np.uint8), np.zeros((used_height, used_width), dtype=np.uint8), np.zeros((used_height, used_width), dtype=np.uint8)), axis=2)
                    # Update visible background statistics
                    visible_backbround_series[frame_ind] = 1.0 - (tracker.background_msk.mean() / 255)
                    # Plot data.
                    fig, (ax_area, ax_drop) = plt.subplots(2, 1, dpi=300, figsize=(2 * used_width / 300, used_height / 300))
                    try:
                        # Background area.
                        ax_area.plot(time_series, 100*visible_backbround_series)
                        ax_area.set_title('Clean Background')
                        ax_area.axes.set_xlabel('Time (seconds)')
                        ax_area.axes.set_ylabel('Area (%)')
                        ax_area.axes.set_ylim(0.0, 105.0)
                        ax_area.grid(True)
                        # Drop.
                        max_id = 0
                        for drop in tracker.drops:
                            id = drop['id']
                            plt.plot((time_series[drop['first_seen']], time_series[drop['last_seen']]), (id, id), marker='x', color=DROP_ID_CMAP[id % len(DROP_ID_CMAP), :] / 256)
                            max_id = max(max_id, id)
                        ax_drop.set_title('Detected Drops')
                        ax_drop.axes.set_xlabel('Time (seconds)')
                        ## ax_drop.axes.set_xlim(*ax_area.axes.get_xlim())
                        ax_drop.axes.set_ylabel('Drop')
                        ax_drop.axes.set_ylim(0, max_id + 1)
                        ax_drop.axes.set_yticks(np.arange(max_id + 1))
                        ax_drop.axes.set_yticklabels(['', *map(str, range(1, max_id + 1))])
                        ax_drop.grid(True)
                        # Draw plots.
                        plt.tight_layout()
                        fig.canvas.draw()
                        analysis_rgb = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
                    finally:
                        plt.close(fig)
                        gc.collect()
                    # Write data.
                    out.write(np.concatenate((frame_rgb, oil_spill_rgb, analysis_rgb), axis=1))
                    # Keep previous frame.
                    previous_frame_rgb = frame_rgb
                    previous_frame_gray = frame_gray
                    previous_oil_spill = oil_spill


if __name__ == '__main__':
    main()
