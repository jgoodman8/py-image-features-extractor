import pyspark.ml.linalg as ml
from pyspark.sql.types import *


class FeatureBuilder:

    def __init__(self, spark, k, model):
        self.k = 500
        self.cluster_model = model

        self.spark = spark
        self.cluster_bins = range(k)

        self.schema = StructType([
            StructField('features', ml.VectorUDT()),
            StructField('label', IntegerType())
        ])

    def get_features_from_descriptors(self, instance):
        """
        For a given instance, finds out what is the cluster related to every descriptor. And returns a list with the
        number of descriptors that contains each cluster (features).

        :param instance: An image instance with label and descriptors
        :return: Tuple of dense vector features and label
        """

        label = instance[0][1]
        instance_df = self.spark.createDataFrame(instance, self.schema)

        cluster_predictions = self.cluster_model.transform(instance_df)

        predictions_column = cluster_predictions.select("prediction")
        predictions = [int(row.prediction) for row in predictions_column.collect()]

        frequencies = self._get_cluster_frequency(predictions)

        return ml.DenseVector(frequencies), label

    def _get_cluster_frequency(self, predictions):
        """
        Calculates the number of predictions that belongs to each cluster.

        :param predictions: Cluster number prediction over each image descriptor within a single instance
        :return: List with the number of descriptors that contains each cluster
        """

        return list(map(lambda bins: predictions.count(bins), self.cluster_bins))
