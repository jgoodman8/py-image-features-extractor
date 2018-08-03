import os
from pathlib import Path

from pyspark.sql.functions import lit
from pyspark.ml.image import ImageSchema


class ImageUtils:

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

        return ImageSchema.readImages(path).withColumn('label', lit(label))
