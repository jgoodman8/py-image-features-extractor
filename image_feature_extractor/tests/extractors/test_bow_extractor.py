import pytest

from image_feature_extractor.extractors.bow_extractor import BoWExtractor
from image_feature_extractor.tests import utils as test_utils


class TestBowExtractor(object):
    
    @pytest.mark.parametrize('method', ['kaze'])
    def test_keypoints_and_descriptors_detection(self, method: str):
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        descriptors, keypoints = extractor._process_image(image_route=test_utils.get_test_image_route())
        
        assert (len(descriptors) > 0)
        assert (keypoints.shape[0] == len(descriptors))
        assert (keypoints.shape[1] == 64)
    
    @pytest.mark.parametrize('method', ['kaze'])
    def test_feature_detection_and_addition(self, method: str):
        image_route = test_utils.get_test_image_route()
        
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        extractor._extract_and_add_image(image_route)
        
        descriptors = extractor.bow_kmeans.getDescriptors()
        expected_keypoints, expected_descriptors = extractor._process_image(image_route)
        
        assert (len(descriptors[0]) == len(expected_descriptors))
        assert (len(descriptors[0]) == len(expected_keypoints))
    
    @pytest.mark.parametrize('method', ['kaze'])
    def test_extraction_setup(self, method: str):
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        extractor.setup()
        
        descriptors = extractor.bow_kmeans.getDescriptors()
        
        assert (len(descriptors) > 0)
        assert (len(descriptors) <= len(extractor.image_keypoints.keys()))
    
    @pytest.mark.parametrize('method', ['kaze'])
    def test_features_are_extracted_from_a_given_image_route(self, method: str):
        output_file = test_utils.get_test_output_csv_route()
        extractor = BoWExtractor(base_route=test_utils.get_test_base_route(), k=3, method=method, size=64)
        extractor.setup()
        extractor.fit()
        
        extractor.extract_and_save(output_file)
        
        features = test_utils.load_csv_from_route(output_file)
        
        expected_width = extractor.k + 1
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
