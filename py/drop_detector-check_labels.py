import matplotlib
matplotlib.use('TkAgg')

from PIL import Image, ImageDraw
from typing import List, Sequence
import argparse, os, random
import default_arguments
import numpy as np
import matplotlib.pyplot as plt


def plot_bounding_box(image: Image, label_list: List[Sequence[float]]) -> None:
    labels = np.array(label_list)
    width, height = image.size
    plotted_image = ImageDraw.Draw(image)
    transformed_labels = np.copy(labels)
    transformed_labels[:, [1, 3]] = labels[:, [1, 3]] * width
    transformed_labels[:, [2, 4]] = labels[:, [2, 4]] * height
    transformed_labels[:, 1] = transformed_labels[:, 1] - (transformed_labels[:, 3] / 2)
    transformed_labels[:, 2] = transformed_labels[:, 2] - (transformed_labels[:, 4] / 2)
    transformed_labels[:, 3] = transformed_labels[:, 1] + transformed_labels[:, 3]
    transformed_labels[:, 4] = transformed_labels[:, 2] + transformed_labels[:, 4]
    for _, x0, y0, x1, y1 in transformed_labels:
        plotted_image.rectangle(((x0,y0), (x1,y1)))
    plt.imshow(np.array(image))
    plt.show()


def main(args: argparse.Namespace) -> None:
    labels_dir = os.path.join(args.dataset, 'labels')
    images_dir = os.path.join(args.dataset, 'images')
    # Get any random label file.
    label_file = random.choice([os.path.join(labels_dir, filename) for filename in os.listdir(labels_dir) if filename.endswith('.txt')])
    with open(label_file, 'r') as file:
        label_list = [list(map(float, line[:-1].split())) for line in file.readlines()]
    # Get the corresponding image.
    image = Image.open(os.path.join(images_dir, f'{os.path.splitext(os.path.basename(label_file))[0]}.png'))
    #Plot the Bounding Box
    plot_bounding_box(image, label_list)


if __name__ == '__main__':
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(*default_arguments.DATASET_ARGS, **default_arguments.DATASET_KWARGS)
    # Parse arguments.
    args = parser.parse_args()
    # Call the main method.
    main(args)
