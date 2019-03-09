from setuptools import find_packages, setup

import image_feature_extractor

setup(
    name='py-image-feature-extractor',
    version=image_feature_extractor.__version__,
    author='Javier Guzman',
    author_email='jguzmanfd@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': ['image_feature_extractor = image_feature_extractor.cli:main']},
    setup_requires=[
        'click',
        'numpy',
        'tensorflow',
        'pandas',
    ],
    url='https://github.com/jgoodman8/py-image-feature-selector',
)
