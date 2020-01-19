import os
import shutil
import sys
from typing import List, Dict

import click
import numpy as np
import scipy.io


@click.command()
@click.option('--src', type=str, required=True, help='Folder where all the images are stored')
@click.option('--dst', type=str, required=True, help='Folder to store the images with the new scaffolding')
@click.option('--descriptor', type=str, required=True, help='File where image labels are stored')
@click.option('--splits', type=str, required=True, help='File where data splits are stored')
def run(src: str, dst: str, descriptor: str, splits: str):
    labels = load_labels(descriptor)
    splits = load_splits(splits)
    images = get_images_routes(src)
    images.sort()
    
    for index, (image, label) in enumerate(zip(images, labels), 1):
        if index in splits['train']:
            copy(image_file=image, dst=dst, label=label, split='train')
        elif index in splits['val']:
            copy(image_file=image, dst=dst, label=label, split='val')
        elif index in splits['test']:
            copy(image_file=image, dst=dst, label=label, split='test')
        else:
            print(f'Error: Image {image} not copied')


def copy(image_file: str, dst: str, label: int, split: str):
    dst_path: str = build_dst(file=image_file, dst=dst, label=str(label), split=split)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # print(f'Copying {image_file} ==> {dst_path}')
    shutil.copyfile(image_file, dst_path)


def build_dst(file: str, dst: str, label: str, split: str) -> str:
    return os.path.join(dst, split, label, file.split('/')[-1])


def load_labels(descriptor: str) -> np.ndarray:
    labels_descriptor = scipy.io.loadmat(descriptor)
    if labels_descriptor is not None:
        return labels_descriptor['labels'][0]
    
    sys.exit("The labels descriptor file doesn't contain the labels")


def load_splits(splits: str) -> Dict:
    splits_descriptor = scipy.io.loadmat(splits)
    if splits_descriptor is not None:
        return {
            'train': splits_descriptor['trnid'][0],
            'val': splits_descriptor['valid'][0],
            'test': splits_descriptor['tstid'][0],
        }
    
    sys.exit("The labels descriptor file doesn't contain the splits")


def get_images_routes(src: str) -> List[str]:
    return [os.path.join(src, file) for file in os.listdir(src)]


if __name__ == '__main__':
    run()
