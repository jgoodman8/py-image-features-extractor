import sys

from . import image_model

data_route = sys.argv[1]

extractor = image_model.ImageModel(data_route)
metrics = extractor.train()

print(metrics)
