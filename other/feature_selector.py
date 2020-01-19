import os
from pyspark.sql import SparkSession

from other.model import Model
from other.image_utils import ImageUtils
from other.feature_builder import FeatureBuilder
from other.feature_extractor import FeatureExtractor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

extraction_algorithm = FeatureExtractor.SIFT

if __name__ == '__main__':
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'

    basePath = 'tiny-imagenet-200'

    spark = SparkSession.builder \
        .master('local[*]') \
        .appName('ImageFeatureSelector') \
        .config('spark.executor.memory', '1G') \
        .config('spark.cores.max', '4') \
        .getOrCreate()

    # Load data
    image_utils = ImageUtils(basePath, spark)
    train = image_utils.load_train_data_as_matrix()
    test = image_utils.load_test_data_as_matrix()

    # Feature detection

    bag_of_descriptors_rdd = train \
        .flatMap(FeatureExtractor.extract_descriptors_from_image) \
        .filter(lambda item: len(item[0]) > 0)

    bow_model = Model(train, None, spark)
    k_means_model = bow_model.create_k_means_model()

    train_with_descriptors_rdd = train \
        .map(FeatureExtractor.extract_descriptors_from_image) \
        .filter(lambda row: row is not None)
    train_with_descriptors = train_with_descriptors_rdd.collect()

    test_with_descriptors_rdd = test \
        .map(FeatureExtractor.extract_descriptors_from_image) \
        .filter(lambda row: row is not None)
    test_with_descriptors = test_with_descriptors_rdd.collect()

    # Feature creation
    feature_builder = FeatureBuilder(spark, bow_model.k, k_means_model)

    train_with_features = map(feature_builder.get_features_from_descriptors, train_with_descriptors)
    train_with_features_df = spark.createDataFrame(train_with_features, feature_builder.schema)

    test_with_features = map(feature_builder.get_features_from_descriptors, test_with_descriptors)
    test_with_features_df = spark.createDataFrame(test_with_features, feature_builder.schema)

    # Train and predict
    main_model = Model(train_with_features, test_with_features_df, spark)
    main_model.index_data()
    main_model.build_decision_tree_model()

    predictions = main_model.model.transform(main_model.test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabels", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print("Test Error = %g " % (1.0 - accuracy))
