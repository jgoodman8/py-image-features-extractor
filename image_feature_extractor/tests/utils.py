import os
import random

import pandas as pd


def get_test_base_route() -> str:
    return 'micro-imagenet/train'


def get_test_image_route() -> str:
    base_route = get_test_base_route()
    image_idx = random.randint(0, 499)
    images_folder = os.path.join(base_route, os.listdir(base_route)[0], 'images')
    return os.path.join(images_folder, os.listdir(images_folder)[image_idx])


def get_test_output_csv_route() -> str:
    return 'output/features.csv'


def clean_test_output_csv_route() -> None:
    route = get_test_output_csv_route()
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
