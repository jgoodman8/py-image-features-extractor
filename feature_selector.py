import os

from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from sparkdl import DeepImageFeaturizer

from image_utils import ImageUtils

if __name__ == '__main__':
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
    print(os.getenv("JAVA_HOME"))

    SparkContext.setSystemProperty('spark.executor.memory', '700M')
    SparkContext.setSystemProperty('spark.cores.max', '4')
    sc = SparkContext(master='spark://localhost:7077', appName='ImageFeatureSelector')
    sc.setLogLevel('INFO')

    basePath = 'micro-imagenet'
    train = ImageUtils.load_train_data(basePath)

    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])

    p_model = p.fit(train)

    print(p_model)
