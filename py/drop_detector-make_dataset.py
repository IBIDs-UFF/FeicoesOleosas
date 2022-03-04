import matplotlib
matplotlib.use('TkAgg')

from typing import List, Optional, Tuple
import argparse, os
import cv2
import default_arguments, utils
import numpy as np


bboxes: List[Tuple[int, int, int, int]] = []
drawing = False
frame_bgr: Optional[np.ndarray] = None
ix, iy = 0, 0


def draw(event: int, x: int, y: int, *_) -> None:
    global bboxes, drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            local_frame_bgr = frame_bgr.copy()
            for bbox in bboxes:
                cv2.rectangle(local_frame_bgr, pt1=(bbox[0], bbox[1]), pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0, 255, 0), thickness=1)
            cv2.rectangle(local_frame_bgr, pt1=(ix, iy), pt2=(x, y), color=(255, 0, 255), thickness=1)
            cv2.imshow('Video', local_frame_bgr)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        local_frame_bgr = frame_bgr.copy()
        for bbox in bboxes:
            cv2.rectangle(local_frame_bgr, pt1=(bbox[0], bbox[1]), pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0, 255, 0), thickness=1)
        cv2.rectangle(local_frame_bgr, pt1=(ix, iy), pt2=(x, y), color=(255, 0, 255), thickness=1)
        cv2.imshow('Video', local_frame_bgr)
        bboxes.append((min(ix, x), min(iy, y), abs(ix - x), abs(iy - y)))


def main(args: argparse.Namespace) -> None:
    global bboxes, frame_bgr
    labels_dir = os.path.join(args.dataset, 'labels')
    images_dir = os.path.join(args.dataset, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # Create the video capture object and open the input file.
    video_in = cv2.VideoCapture(args.input)
    try:
        # Initialize som useful variables.
        frame_ind = 0
        num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video_in.get(cv2.CAP_PROP_FPS))
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        used_width, used_height = default_arguments.compute_used_size(width, height, args.frame_size)
        # Create a window.
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Video', draw)
        # For each frame...
        ret, frame_bgr = video_in.read()
        while ret:
            frame_bgr = cv2.resize(frame_bgr, (used_width, used_height))
            # Show current frame to the user.
            bboxes.clear()
            cv2.setWindowTitle('Video', f'Frame {frame_ind + 1}/{num_frames}')
            cv2.imshow('Video', frame_bgr)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('+'):
                frame_ind += 1
                ret, frame_bgr = video_in.read()
                skip = fps
                while ret and skip > 0:
                    frame_ind += 1
                    skip -= 1
                    ret, frame_bgr = video_in.read()
            if key == 13:
                if len(bboxes) > 0:
                    mp4_name = os.path.splitext(os.path.basename(args.input))[0]
                    cv2.imwrite(os.path.join(images_dir, f'{mp4_name}-frame_{frame_ind + 1}.png'), frame_bgr)
                    with open(os.path.join(labels_dir, f'{mp4_name}-frame_{frame_ind + 1}.txt'), 'w') as txt:
                        for x, y, w, h in bboxes:
                            if w > 0 and h > 0:
                                txt.write(f'0 {(x + (w / 2)) / used_width} {(y + (h / 2)) / used_height} {w / used_width} {h / used_height}\n')
            elif key == 27:
                break
            # Go to the next frame.
            frame_ind += 1
            ret, frame_bgr = video_in.read()
        # Destroy the window.
        cv2.destroyAllWindows()
    finally:
        video_in.release()


if __name__ == '__main__':
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(*default_arguments.INPUT_ARGS, **default_arguments.INPUT_KWARGS)
    parser.add_argument(*default_arguments.DATASET_ARGS, **default_arguments.DATASET_KWARGS)
    parser.add_argument(*default_arguments.FRAME_SIZE_ARGS, **default_arguments.FRAME_SIZE_KWARGS)
    # Parse arguments.
    args = parser.parse_args()
    # Call the main method.
    main(args)
