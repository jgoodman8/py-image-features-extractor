import os
import random

import pandas as pd
from click.testing import Result


def get_test_base_route() -> str:
    return 'micro-imagenet/train'


def get_test_output_csv_route() -> str:
    return 'output/features.csv'


def get_test_output_ndarray_route() -> str:
    return 'output/vocabulary.npy'


def get_test_image_route() -> str:
    base_route = get_test_base_route()
    image_idx = random.randint(0, 499)
    images_folder = os.path.join(base_route, os.listdir(base_route)[0], 'images')
    return os.path.join(images_folder, os.listdir(images_folder)[image_idx])


def clean_test_output_csv_route() -> None:
    route = get_test_output_csv_route()
    if os.path.exists(route):
        os.remove(route)


def clean_test_output_ndarray_route() -> None:
    route = get_test_output_ndarray_route()
    if os.path.exists(route):
        os.remove(route)


def load_csv_from_route(route: str) -> pd.DataFrame:
    return pd.read_csv(route)


def count_test_images() -> int:
    num_images = 0
    
    base_route = get_test_base_route()
    for folder in os.listdir(base_route):
        num_images += len(os.listdir(os.path.join(base_route, folder, 'images')))
    
    return num_images


def assert_validation_test(result: Result, output_file: str = get_test_output_csv_route()):
    assert (result.exit_code == 0)
    
    output = load_csv_from_route(output_file)
    assert (output.size > 0)


def get_expected_descriptor_size(descriptor: str) -> int:
    descriptors_sizes = {'akaze': 61, 'kaze': 64, 'orb': 32, 'brisk': 64, 'sift': 128, 'surf': 64}
    return descriptors_sizes[descriptor]
