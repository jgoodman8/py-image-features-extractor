from setuptools import find_packages, setup

long_description = '''
This package provides implementations of different methods to perform image feature extraction. These methods are
though a Python package and a command line interface. Available feature extraction methods are:

- Convolutional Neural Networks
  
  - VGG-19
  - ResNet-50
  - DenseNet-50
  - Custom CNN through .h5 file

- Linear Binary Patterns Histograms (LBPH)

- Bag of Features (bag-of-visual-words)

  - SIFT
  - SURF
  - KAZE
'''

setup(
    name='py-image-feature-extractor',
    version='0.1.1',
    author='Javier Guzman',
    author_email='jguzmanfd@gmail.com',
    description='This package provides implementations of different methods to perform image feature extraction',
    long_description=long_description,
    packages=find_packages(),
    entry_points={'console_scripts': ['image_feature_extractor = image_feature_extractor.cli:main']},
    install_requires=[
        'click',
        'numpy',
        'scikit-image',
        'tensorflow',
        'opencv-python==3.4.2.17',
        'opencv-contrib-python==3.4.2.17',
        'pandas',
        'pytest',
        'matplotlib',
        'scikit-learn',
        'scipy',
    ],
    url='https://github.com/jgoodman8/py-image-feature-selector',
)
