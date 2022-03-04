from tqdm import tqdm
from typing import List, Tuple
import argparse, os, shutil, subprocess, sys
import default_arguments
import oil
import sklearn.model_selection


DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_WORKERS = 4


def split_dataset(dataset_dir: str) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    # Read images.
    images_dir = os.path.join(dataset_dir, 'images')
    images = [os.path.join(images_dir, filename) for filename in os.listdir(images_dir) if filename.endswith('.png')]
    images.sort()
    # Read labels.
    labels_dir = os.path.join(dataset_dir, 'labels')
    labels = [os.path.join(labels_dir, filename) for filename in os.listdir(labels_dir) if filename.endswith('.txt')]
    labels.sort()
    # Split the dataset into train-test subsets.
    train_images, val_images, train_labels, val_labels = sklearn.model_selection.train_test_split(images, labels, test_size=0.2)
    val_images, test_images, val_labels, test_labels = sklearn.model_selection.train_test_split(val_images, val_labels, test_size=0.5)
    # Return splits.
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def main(args: argparse.Namespace) -> None:
    # Set the path to key folders.
    dataset_split_dir = os.path.join(os.path.dirname(__file__), '.drop_dataset')
    yolov5_dir = os.path.join(os.path.dirname(__file__), 'oil', 'yolov5')
    if os.path.isdir(dataset_split_dir):
        shutil.rmtree(dataset_split_dir)
    os.makedirs(dataset_split_dir)
    # Copy the dataset to split folders.
    for (image_files, label_files), subset_name in zip(split_dataset(args.dataset), ['train', 'val', 'test']):
        images_dir = os.path.join(dataset_split_dir, 'images', subset_name)
        os.makedirs(images_dir)
        for original_file in tqdm(image_files, desc=f'Copying {subset_name} image files'):
            shutil.copyfile(original_file, os.path.join(images_dir, os.path.basename(original_file)))
        labels_dir = os.path.join(dataset_split_dir, 'labels', subset_name)
        os.makedirs(labels_dir)
        for original_file in tqdm(label_files, desc=f'Copying {subset_name} label files'):
            shutil.copyfile(original_file, os.path.join(labels_dir, os.path.basename(original_file)))
    # Call YOLOv5's train script.
    train_script_path = os.path.join(yolov5_dir, 'train.py')
    subprocess.run([sys.executable, train_script_path, '--img', str(args.frame_size), '--cfg', 'drop_yolov5s.yaml', '--hyp', 'hyp.scratch-low.yaml', '--batch', str(args.batch_size), '--epochs', str(args.epochs), '--data', 'drop_data.yaml', '--weights', 'yolov5s.pt', '--workers', str(args.workers), '--name', 'yolo_drop_det'])
    # Copy best trained model to the YOLOv5's root folder.
    shutil.copyfile(os.path.join(yolov5_dir, 'runs', 'train', 'yolo_drop_det', 'weights', 'best.pt'), oil.DROP_DETECTOR_WEIGHTS_FILEPATH)


if __name__ == '__main__':
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(*default_arguments.DATASET_ARGS, **default_arguments.DATASET_KWARGS)
    parser.add_argument(*default_arguments.FRAME_SIZE_ARGS, **default_arguments.FRAME_SIZE_KWARGS)
    parser.add_argument('--batch_size', metavar='SIZE', type=int, default=DEFAULT_BATCH_SIZE, help='the batch size')
    parser.add_argument('--epochs', metavar='NUMBER', type=int, default=DEFAULT_EPOCHS, help='number of epochs to train for')
    parser.add_argument('--workers', metavar='NUMBER', type=int, default=DEFAULT_WORKERS, help='number of workers')
    # Parse arguments.
    args = parser.parse_args()
    # Call the main method.
    main(args)
