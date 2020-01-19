import os
import numpy as np
import pytest

from image_feature_extractor.extractors.bow_extractor import BoWExtractor
from image_feature_extractor.tests import utils as test_utils


class TestBowExtractor(object):
    
    @pytest.fixture(autouse=True)
    def clear_output_folder(self):
        # No actions before every test
        yield
        test_utils.clean_test_output_csv_route()
        test_utils.clean_test_output_ndarray_route()
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_keypoints_and_descriptors_detection(self, method: str):
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        descriptors, keypoints = extractor._process_image(image_route=test_utils.get_test_image_route())
        
        assert (len(descriptors) > 0)
        assert (keypoints.shape[0] == len(descriptors))
        assert (keypoints.shape[1] == extractor.detector.descriptorSize())
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_feature_detection_and_addition(self, method: str):
        image_route = test_utils.get_test_image_route()
        
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        extractor._extract_and_add_image(image_route)
        
        descriptors = extractor.bow_kmeans.getDescriptors()
        expected_keypoints, expected_descriptors = extractor._process_image(image_route)
        
        assert (len(descriptors[0]) == len(expected_descriptors))
        assert (len(descriptors[0]) == len(expected_keypoints))
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_extraction_setup(self, method: str):
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        extractor.setup()
        
        descriptors = extractor.bow_kmeans.getDescriptors()
        
        assert (len(descriptors) > 0)
        assert (len(descriptors) <= len(extractor.image_keypoints.keys()))
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_features_are_extracted_from_a_given_image_route_in_manual_mode(self, method: str):
        output_file = test_utils.get_test_output_csv_route()
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64,
                                 cluster_mode='manual')
        extractor.setup()
        extractor.fit()
        
        extractor.extract_and_save(output_file)
        
        features = test_utils.load_csv_from_route(output_file)
        
        expected_width = extractor.k + 1
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_kmeans_model_is_exported(self, method: str):
        vocabulary_file = test_utils.get_test_output_ndarray_route()
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64,
                                 cluster_mode='manual')
        extractor.setup()
        extractor.fit()
        
        extractor.export(vocabulary_file)
        
        assert (os.path.exists(vocabulary_file))
        vocabulary = np.load(vocabulary_file)
        assert (vocabulary.shape[0] == extractor.k)
        assert (np.array_equal(vocabulary, extractor.bow_extractor.getVocabulary()))
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_kmeans_model_is_loaded_from_file(self, method: str):
        vocabulary_file = test_utils.get_test_output_ndarray_route()
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64,
                                 cluster_mode='manual')
        extractor.setup()
        extractor.fit()
        extractor.export(vocabulary_file)
        
        new_extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64,
                                     cluster_mode='manual')
        new_extractor.setup()
        new_extractor.load(vocabulary_file)
        
        assert (np.array_equal(new_extractor.bow_extractor.getVocabulary(), extractor.bow_extractor.getVocabulary()))
    
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_features_are_extracted_using_a_loaded_model(self, method: str):
        output_file = test_utils.get_test_output_csv_route()
        vocabulary_file = test_utils.get_test_output_ndarray_route()
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64,
                                 cluster_mode='manual')
        extractor.setup()
        extractor.fit()
        extractor.export(vocabulary_file)
        
        new_extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64,
                                     cluster_mode='manual')
        new_extractor.setup()
        new_extractor.load(vocabulary_file)
        new_extractor.extract_and_save(output_file)
        
        features = test_utils.load_csv_from_route(output_file)
        
        expected_width = extractor.k + 1
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
    
    @pytest.mark.skip(reason="Very slow test to run always")
    @pytest.mark.parametrize('method', ['kaze', 'sift', 'surf'])
    def test_features_are_extracted_from_a_given_image_route_in_automatic_mode(self, method: str):
        output_file = test_utils.get_test_output_csv_route()
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), method=method, size=64,
                                 cluster_mode='auto', min_k=2, max_k=100, step=1, threshold=0.95)
        extractor.setup()
        extractor.fit()
        
        extractor.extract_and_save(output_file)
        
        features = test_utils.load_csv_from_route(output_file)
        
        expected_width = extractor.k + 1
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
