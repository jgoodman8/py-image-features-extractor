import sys
from cnn_feature_extractor import FeatureExtractor

data_route = sys.argv[1]
model_name = sys.argv[2]
output= sys.argv[3]

extractor = FeatureExtractor(data_route, model_name, output)
extractor.extract()
