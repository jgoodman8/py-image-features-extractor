import sys

from cnn.models.image_model import ImageModel

data_route = sys.argv[1]
model_name = sys.argv[2]

extractor = ImageModel(data_route)
metrics = extractor.train()

print(metrics)
