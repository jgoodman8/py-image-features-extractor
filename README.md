# Py Image Feature Extractor 

## Index

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Tests](#tests)
5. [Run](#run)

## <a name="overview">Overview</a>

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
  
At the `notebooks` folder, some proofs-of-concept related to feature extraction and image classification may be found.

## <a name="requirements">Requirements</a>

System requirements:

- python >= 3.7.3
- pip >= 19.1.1

All the package requirements are listed on the `install_requires` property within the `setup.py`. 

## <a name="installation">Installation</a>

This project may be installed as a python package using:

```bash
pip install .
```

## <a name="tests">Tests</a>

All the test suite has been developed using the [pytest framework](https://docs.pytest.org/en/latest/).

```bash
# All tests
pytest

# Unit tests of extractors module
pytest image_feature_extractor/tests/extractors

# Unit tests of models module
pytest image_feature_extractor/tests/models

# Validation tests
pytest image_feature_extractor/tests/validation
```

## <a name="run">Run</a>

### Model

The package has a command-line entry point configured. This entry point is built using the library 
[Click](https://palletsprojects.com/p/click/). To get all the possible commands, use `image_feature_extractor --help`.

```bash
# Example to perform feature extraction using a pre-trained VGG-19
image_feature_extractor extract --deep --src imgs/train --dst vgg19_train.csv --cnn vgg19 --size 200

# Example to perform feature extraction using LBPs
image_feature_extractor extract --lbp --src imgs/train --dst vgg19_train.csv --detector kaze vgg19 --k 100 --size 200 --export --vocabulary-route vocabulary.npy

# Example to perform feature extraction using bag-of-features with KAZE keypoint detector
image_feature_extractor extract --bow --src imgs/train --dst vgg19_train.csv --points 8 --radius 1 --grid 8 --size 200
```
