import os
import PIL.Image
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sparkdl.estimators.keras_image_file_estimator import KerasImageFileEstimator

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'


def load_data(folder):
    """
    Creates a list of dictionaries that contains all the labels and image routes

    :param folder: The train data path
    :return: A list of dictionaries with properties: 'uri' and 'one_hot_label'.
    """

    data = []

    image_class_folders = [sub_folder for sub_folder in folder.iterdir() if sub_folder.is_dir()]

    for class_folder in image_class_folders:
        images_folder = Path(str(class_folder) + '/images')

        images = [image_dir for image_dir in images_folder.iterdir()]

        for image in images:
            data.append({
                'one_hot_label': int(str(image).split('/').pop().split('_')[0][1:]),
                'uri': str(image)
            })

    return data


def load_train_data(base_path):
    """
    Gets the train data from the given base path

    :param base_path: Imagenet base path
    :return: A list of dictionaries with properties: 'uri' and 'one_hot_label'.
    """

    return load_data(Path(base_path + '/train'))


def load_image_from_uri(local_uri):
    """
    Returns a given image processed, for a given image route.

    :param local_uri: Full image route
    :return: Preprocessed tensor or Numpy array.
    """

    img = (PIL.Image.open(local_uri).convert('RGB').resize((299, 299), PIL.Image.ANTIALIAS))
    img_arr = np.array(img).astype(np.float32)
    img_tnsr = preprocess_input(img_arr[np.newaxis, :])
    return img_tnsr


if __name__ == '__main__':
    imagenet_path = 'tiny-imagenet-200'

    spark = SparkSession.builder \
        .master('local[2]') \
        .appName('ImageFeatureSelector') \
        .config('spark.executor.memory', '2G') \
        .config('spark.executor.cores', '2') \
        .config('spark.driver.memory', '3G') \
        .config('spark.driver.cores', '1') \
        .getOrCreate()

    train_df = spark.createDataFrame(load_train_data(imagenet_path))

    pre_trained_model = InceptionV3(weights="imagenet")
    pre_trained_model.save('/tmp/model-full.h5')

    estimator = KerasImageFileEstimator(inputCol="uri",
                                        outputCol="prediction",
                                        labelCol="one_hot_label",
                                        imageLoader=load_image_from_uri,
                                        kerasOptimizer='adam',
                                        kerasLoss='categorical_crossentropy',
                                        modelFile='/tmp/model-full-tmp.h5'  # local file path for model
                                        )

    param_grid = (ParamGridBuilder().addGrid(estimator.kerasFitParams, [{"batch_size": 32, "verbose": 0},
                                                                        {"batch_size": 64, "verbose": 0}]).build())

    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
    cv = CrossValidator(estimator=estimator, estimatorParamMaps=param_grid, evaluator=binary_evaluator, numFolds=2)

    cv_model = cv.fit(train_df)

    print(cv_model)
