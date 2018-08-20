import os

from pyspark import SparkContext

from feature_extractor import FeatureExtractor
from image_utils import ImageUtils

if __name__ == '__main__':
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
    print(os.getenv("JAVA_HOME"))

    SparkContext.setSystemProperty('spark.executor.memory', '1G')
    SparkContext.setSystemProperty('spark.cores.max', '4')
    sc = SparkContext(master='local[*]', appName='ImageFeatureSelector')
    sc.setLogLevel('INFO')

    basePath = 'tiny-imagenet-200'

    image_utils = ImageUtils(basePath, sc)
    train = image_utils.load_train_data_as_matrix()

    feature_extractor = FeatureExtractor(train)
    train_with_features = feature_extractor.extract_features()


