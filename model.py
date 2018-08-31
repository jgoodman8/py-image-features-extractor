import pyspark.ml.linalg as ml
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import *


class Model:

    def __init__(self, train, test, spark):
        schema = StructType([
            StructField('features', ml.VectorUDT()),
            StructField('label', IntegerType())
        ])

        self.train_df = spark.createDataFrame(train, schema) if train is not None else None
        self.test_df = spark.createDataFrame(test, schema) if test is not None else None

        self.model = None
        self.k = 40

    def create_k_means_model(self):
        """
        Fits a kMeans model for the given training set

        :return: A kMeans model
        """
        model_route = "hdfs://namenode/sift_descriptors_kmeans_40"

        k_means = KMeans().setK(self.k).setSeed(3)
        self.model = k_means.fit(self.train_df)
        self.model.save(model_route)

        return self.model

    def build_logistic_regression_model(self):
        logistic_regression = LogisticRegression()
        self.model = logistic_regression.fit(self.train_df)

    def predict(self):
        return self.model.predict(self.test_df)

    def index_data(self):
        indexer = StringIndexer(inputCol="label", outputCol="indexedLabels")

        self.train_df = indexer.fit(self.train_df).transform(self.train_df)
        self.test_df = indexer.fit(self.train_df).transform(self.test_df)

    def build_decision_tree_model(self):
        decision_tree = DecisionTreeClassifier(labelCol="indexedLabels", featuresCol="features")
        self.model = decision_tree.fit(self.train_df)
