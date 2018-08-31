import cv2

import pyspark.ml.linalg as ml

kaze = cv2.KAZE_create()
sift = cv2.xfeatures2d.SIFT_create()


class FeatureExtractor:

    def __init__(self, train, test, spark, k, model):
        self.train = train
        self.test = test

        self.train_with_features = None
        self.test_with_features = None

    @staticmethod
    def perform_sift_extraction(data):
        label = data[0]
        image = data[1]

        key_points, descriptors = sift.detectAndCompute(image, None)

        return label, descriptors

    @staticmethod
    def get_labels(train_with_descriptors):
        return list(map(lambda item: item[0], train_with_descriptors))

    @staticmethod
    def get_descriptors(train_with_descriptors):
        return list(map(lambda item: item[1], train_with_descriptors))

    @staticmethod
    def extract_features_from_data(data):
        return data[0], ml.DenseVector(FeatureExtractor.extract_features_from_image(data[1]).tolist())

    def extract_features(self):
        self.train_with_features = self.train.map(FeatureExtractor.extract_features_from_data)
        self.test_with_features = self.test.map(FeatureExtractor.extract_features_from_data)

    @staticmethod
    def get_extraction_algorithm():
        return sift, 128

    @staticmethod
    def extract_features_from_image(data):
        label = data[0]
        image = data[1]

        algorithm = cv2.xfeatures2d.SIFT_create()

        key_points, descriptors = algorithm.detectAndCompute(image, None)

        if descriptors is None:
            return None

        dense_descriptors = list(map(lambda descriptor: (ml.DenseVector(descriptor.tolist()), label), descriptors))

        return dense_descriptors

    @staticmethod
    def find_key_points_in_image(item):

        algorithm = cv2.xfeatures2d.SIFT_create()
        label = item[0]
        key_points, descriptors = algorithm.detectAndCompute(item[1], None)

        if descriptors is None:
            return list([(ml.DenseVector([]), label)])

        dense_descriptors = list(map(lambda descriptor: (ml.DenseVector(descriptor.tolist()), label), descriptors))

        return dense_descriptors
