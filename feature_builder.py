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

    def get_cluster_frequency(self, predictions):
        return list(map(lambda bin: predictions.count(bin), self.cluster_bins))

    def get_features_from_descriptors(self, instance):
        label = instance[0][1]
        instance_df = self.spark.createDataFrame(instance, self.schema)

        cluster_predictions = self.cluster_model.transform(instance_df)

        predictions_column = cluster_predictions.select("prediction")
        predictions = [int(row.prediction) for row in predictions_column.collect()]

        frequencies = self.get_cluster_frequency(predictions)

        return ml.DenseVector(frequencies), label
