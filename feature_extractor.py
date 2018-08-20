import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
kaze = cv2.KAZE_create()


class FeatureExtractor:

    def __init__(self, train):
        self.train = train

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

    def extract_features(self):
        train_with_features = []
        i = 0

        for instance in self.train:
            instance_with_features = {
                'label': instance[0],
                'features': FeatureExtractor.extract_features_from_image(instance[1])
            }

            train_with_features.append(instance_with_features)

        return train_with_features

    @staticmethod
    def get_extraction_algorithm():
        return sift

    @staticmethod
    def extract_features_from_image(image, vector_size=50):

        algorithm = FeatureExtractor.get_extraction_algorithm()

        # Dinding image keypoints
        key_points = algorithm.detect(image)

        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        key_points = sorted(key_points, key=lambda x: -x.response)[:vector_size]

        # computing descriptors vector
        key_points, descriptors = algorithm.compute(image, key_points)

        if descriptors is None:
            print("debug")
            return np.zeros(vector_size * 64)

        # Flatten all of them in one big vector - our feature vector
        descriptors = descriptors.flatten()

        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)

        if descriptors.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            descriptors = np.concatenate([descriptors, np.zeros(needed_size - descriptors.size)])

        return descriptors
