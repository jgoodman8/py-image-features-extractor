import csv
import os
from pathlib import Path

import cv2
import sh
from pyspark.ml.image import ImageSchema
from pyspark.sql import functions


class ImageUtils:

    def __init__(self, base_path, spark):
        self.base_path = base_path
        self.spark = spark

    @staticmethod
    def load_train_data(base_path):
        train_folder = Path(base_path + '/train')

        return ImageUtils.load_data(train_folder)

    @staticmethod
    def load_test_data(base_path):
        test_folder = Path(base_path + '/train')

        return ImageUtils.load_data(test_folder)

    @staticmethod
    def load_data(folder):
        image_class_folders = [dir for dir in folder.iterdir() if dir.is_dir()]

        data = ImageUtils.create_image_df_with_label(image_class_folders[0])

        for label_index in range(1, len(image_class_folders)):
            data_slice = ImageUtils.create_image_df_with_label(image_class_folders[label_index])
            data = data.unionAll(data_slice)

        return data

    @staticmethod
    def find_images_path(folder):
        return os.path.abspath(folder._str + '/images')

    @staticmethod
    def create_image_df_with_label(image_folder):
        label = int(image_folder.stem[1:])
        path = ImageUtils.find_images_path(image_folder)

        return ImageSchema.readImages(path).withColumn('label', functions.lit(label))

    @staticmethod
    def list_hdfs_content(path):

        return [line.rsplit(None, 1)[-1] for line in sh.hdfs('dfs', '-ls', path).split('\n') if
                len(line.rsplit(None, 1))][1:]

    @staticmethod
    def add_image_labels(image_path):
        """
        Obtains the image label, for a given image route.

        :param image_path: Image route
        :return: Tuple of (route, label)
        """

        label = int(image_path.split('/').pop().split('_')[0][1:])
        return image_path, label

    @staticmethod
    def add_image_matrix(image_tuple):
        """
        Obtains the image data from a given image route

        :param image_tuple: Image route
        :return: N-tuple with (label, image, route)
        """

        route = image_tuple[0]
        label = image_tuple[1]

        image = cv2.imread(route, cv2.IMREAD_COLOR)
        return label, image, route

    @staticmethod
    def get_test_labels(file_route):
        """
        Loads the test labels from the test data set.

        :param file_route: File that contains the labels for each one of the test images.
        :return: List of image labels.
        """

        labels = []

        with open(file_route) as file_descriptor:
            file = csv.reader(file_descriptor, delimiter="\t", quotechar='"')
            for row in file:
                label = int(row[1][1:])
                labels.append(label)

        return labels

    def load_train_data_as_matrix(self):
        """
        Loads the train data from the train folder.

        :return: Spark RDD with tuples of (label, image, route)
        """

        train_path = Path(self.base_path + "/train")

        class_folders = [class_folder for class_folder in train_path.iterdir()]
        image_routes = [str(img) for image_dir in class_folders for img in Path(str(image_dir) + "/images").iterdir()]
        labeled_image_routes = list(map(ImageUtils.add_image_labels, image_routes))

        labeled_image_routes_df = self.spark.createDataFrame(labeled_image_routes)

        return labeled_image_routes_df.rdd.map(ImageUtils.add_image_matrix)

    def load_test_data_as_matrix(self):
        """
        Loads the test data from the train folder.

        :return: Spark RDD with tuples of (label, image, route)
        """

        test_images_path = Path(self.base_path + "/val/images")
        test_annotations_route = self.base_path + "/val/val_annotations.txt"

        test_image_routes = [str(image_path) for image_path in test_images_path.iterdir()]
        test_image_labels = ImageUtils.get_test_labels(test_annotations_route)

        labeled_images_routes = list(zip(test_image_routes, test_image_labels))

        test_labeled_paths_rdd = self.spark.createDataFrame(labeled_images_routes)

        return test_labeled_paths_rdd.rdd.map(ImageUtils.add_image_matrix)
