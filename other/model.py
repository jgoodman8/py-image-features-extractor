import pyspark.ml.linalg as ml
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import *


class Model:
    cluster_model_route = "hdfs://namenode/cluster_model"
    model_route = "hdfs://namenode/cluster_model"

    def __init__(self, train, test, spark):
        schema = StructType([
            StructField('features', ml.VectorUDT()),
            StructField('label', IntegerType())
        ])

        self.k = 40
        self.train_df = spark.createDataFrame(train, schema) if train is not None else None
        self.test_df = spark.createDataFrame(test, schema) if test is not None else None

        self.model = None

    def create_k_means_model(self):
        """
        Fits a kMeans model for the given training set, stores the model at the class property and persist de model
        at the set path

        :return: A trained kMeans model
        """

        k_means = KMeans().setK(self.k).setSeed(3)
        self.model = k_means.fit(self.train_df)
        self.model.save(Model.cluster_model_route)

        return self.model

    def build_logistic_regression_model(self):
        """
        [Not currently used]
        Trains a logistic regression model by using the stored training set, and stores at the the class variable
        """

        logistic_regression = LogisticRegression()
        self.model = logistic_regression.fit(self.train_df)

    def predict(self):
        """
        [Not currently used]
        Builds a prediction given an stored model and test set.

        :return: A DataFrame with the predicted labels for each instance
        """

        return self.model.predict(self.test_df)

    def index_data(self):
        """
        Rewrites the train and test set, in order to replace the current labels with consecutive indexes (which is a
        requirement to fit some models)
        """

        indexer = StringIndexer(inputCol="label", outputCol="indexedLabels")

        self.train_df = indexer.fit(self.train_df).transform(self.train_df)
        self.test_df = indexer.fit(self.train_df).transform(self.test_df)

    def build_decision_tree_model(self):
        """
        Trains a decision tree, by using the pre-indexed train data set, stores it as property and persist it at the
        set path

        :return: A trained decision tree classification model
        """

        decision_tree = DecisionTreeClassifier(labelCol="indexedLabels", featuresCol="features")
        self.model = decision_tree.fit(self.train_df)
        self.model.save(Model.model_route)

        return self.model
