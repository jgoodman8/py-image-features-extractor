from setuptools import find_packages, setup

setup(
    name='py-image-feature-extractor',
    version='0.0.7',
    author='Javier Guzman',
    author_email='jguzmanfd@gmail.com',
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
