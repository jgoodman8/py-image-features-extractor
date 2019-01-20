import sys
from cnn_extractors.feature_extractor import FeatureExtractor

data_route = sys.argv[0]
model_name = sys.argv[1]
output= sys.argv[2]

extractor = FeatureExtractor(data_route, model_name, output)
extractor.extract()
