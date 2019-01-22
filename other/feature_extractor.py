import cv2

import pyspark.ml.linalg as ml

from other.feature_selector import extraction_algorithm


class FeatureExtractor:
    SIFT = "SIFT"
    SURF = "SURF"

    def __init__(self, train, test):
        self.train = train
        self.test = test

        self.train_with_features = None
        self.test_with_features = None

    def extract_features(self):
        """
        Transforms the train and test sets into data sets with extracted descriptors
        """

        self.train_with_features = self.train.map(FeatureExtractor._extract_descriptors_from_data)
        self.test_with_features = self.test.map(FeatureExtractor._extract_descriptors_from_data)

    @staticmethod
    def extract_descriptors_from_image(data):
        """
        Extracts descriptors related to the key-points detected by the used algorithm (SIFT, SURF...)

        :param data: Tuple that contains (label, image data)
        :return: List of dense descriptors (the descriptor length varies depending on the used algorithm) or None if no
                 key points have been detected
        """

        label = data[0]
        image = data[1]

        key_points, descriptors = FeatureExtractor._get_extraction_algorithm().detectAndCompute(image, None)

        if descriptors is None:
            return list([(ml.DenseVector([]), label)])

        dense_descriptors = list(map(lambda descriptor: (ml.DenseVector(descriptor.tolist()), label), descriptors))

        return dense_descriptors

    @staticmethod
    def _extract_descriptors_from_data(data):
        """
        Extracts descriptors from a given data instance

        :param data: A single data instance (label, image)
        :return:
        """

        label = data[0]

        return label, ml.DenseVector(FeatureExtractor.extract_descriptors_from_image(data))

    @staticmethod
    def _get_extraction_algorithm():
        """
        Gets an OpenCV feature detection algorithm instance, depending on the value set on the global variable
        'extraction_algorithm'.

        :return: A feature detection algorithm
        """

        switcher = {
            FeatureExtractor.SIFT: cv2.xfeatures2d.SIFT_create(),
            FeatureExtractor.SURF: cv2.xfeatures2d.SURF_create(),
        }

        return switcher.get(extraction_algorithm)
