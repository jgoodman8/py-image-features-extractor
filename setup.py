from setuptools import setup, find_packages, Command

import cnn
from cnn.extractors.feature_extractor import FeatureExtractor
from cnn.models.image_model import ImageModel


class Train(Command):
  description = 'Train and Evaluate a CNN-based model'
  user_options = [
    ('route=', None, 'Training base route')
  ]
  
  def initialize_options(self):
    self.route = ''
  
  def finalize_options(self):
    pass
  
  def run(self):
    print(self.route)
    
    extractor = ImageModel(self.route)
    metrics = extractor.train()
    
    print(metrics)


class Extract(Command):
  description = 'Extracts features using a CNN-based model'
  user_options = [
    ('route=', None, 'Training base route'),
    ('model=', None, 'Name of the required model (inception_v3, inception_resnet_v2, vgg19)'),
    ('output=', None, 'Route to the output file')
  ]
  
  def initialize_options(self):
    self.route = ''
    self.model = ''
    self.output = ''
  
  def finalize_options(self):
    pass
  
  def run(self):
    extractor = FeatureExtractor(self.data_route, self.model_name, self.output)
    extractor.extract()


setup(
  name='py-image-feature-extractor',
  version=cnn.__version__,
  author='Javier Guzmán',
  author_email='jguzmanfd@gmail.com',
  packages=find_packages(),
  setup_requires=[
    'numpy',
    'tensorflow'
  ],
  url='https://github.com/jgoodman8/py-image-feature-selector',
  cmdclass={
    'train': Train,
    'extract': Extract
  }
)
