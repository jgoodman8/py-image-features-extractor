import os
# from pyspark import SparkContext

from model import Model
from image_utils import ImageUtils
from feature_extractor import FeatureExtractor

if __name__ == '__main__':
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'

    # SparkContext.setSystemProperty('spark.executor.memory', '1G')
    # SparkContext.setSystemProperty('spark.cores.max', '4')
    # sc = SparkContext(master='local[*]', appName='ImageFeatureSelector')
    # sc.setLogLevel('INFO')

    basePath = 'tiny-imagenet-200'

    image_utils = ImageUtils(basePath, None)
    train = image_utils.load_train_data_as_matrix()
    test = image_utils.load_test_data_as_matrix()

    feature_extractor = FeatureExtractor(train, test)
    feature_extractor.extract_features()

    model = Model(feature_extractor.train_with_features, feature_extractor.test_with_features)
    model.build_logistic_regression_model()

    prediction = model.predict()
