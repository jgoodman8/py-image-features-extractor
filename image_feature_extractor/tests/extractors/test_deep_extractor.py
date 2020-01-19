import pytest

from image_feature_extractor.extractors import DeepExtractor
from image_feature_extractor.tests import utils as test_utils


class TestDeepExtractor(object):
    
    @pytest.fixture(autouse=True)
    def clear_output_folder(self):
        # No actions before every test
        yield
        test_utils.clean_test_output_csv_route()
    
    @pytest.mark.parametrize('model', ['vgg19', 'inception_v3', 'inception_resnet_v2'])
    def test_features_are_extracted_from_a_given_image_route(self, model: str):
        extractor = DeepExtractor(base_route=test_utils.get_test_base_route(), model_name=model, size=75)
        
        features = extractor.extract(image_route=test_utils.get_test_image_route())
        
        output_shape = extractor.model.layers[-1].output_shape
        expected_shape = output_shape[1] * output_shape[2] * output_shape[3]
        assert (features.shape[0] == expected_shape)
    
    @pytest.mark.parametrize('model', ['vgg19', 'inception_v3', 'inception_resnet_v2'])
    def test_features_are_extracted_from_a_given_image_route(self, model: str):
        output_csv = test_utils.get_test_output_csv_route()
        
        extractor = DeepExtractor(base_route=test_utils.get_test_base_route(), model_name=model, size=75)
        extractor.extract_and_save(output_csv)
        
        features = test_utils.load_csv_from_route(output_csv)
        
        cnn_output_shape = extractor.model.layers[-1].output_shape
        expected_width = cnn_output_shape[1] * cnn_output_shape[2] * cnn_output_shape[3] + 1
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
