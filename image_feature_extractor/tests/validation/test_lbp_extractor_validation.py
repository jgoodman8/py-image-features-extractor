import pytest

from image_feature_extractor.extractors import LBPExtractor
from image_feature_extractor.tests import utils as test_utils


class TestLBPExtractorValidation(object):
    
    @pytest.fixture(autouse=True)
    def clear_output_folder(self):
        # No actions before every test
        yield
        test_utils.clean_test_output_csv_route()
    
    @pytest.mark.parametrize('x', [2, 4])
    @pytest.mark.parametrize('y', [2, 4])
    def test_(self, x, y):
        output_csv = test_utils.get_test_output_csv_route()
        base_route = test_utils.get_test_images_route()
        
        extractor = LBPExtractor(base_route=base_route, size=64, points=3, radius=1, grid_x=x, grid_y=y)
        extractor.extract_and_save(output_csv)
        
        features = test_utils.load_csv_from_route(output_csv)
        
        expected_width = x * y * (len(extractor.bins) - 1) + 1  # Number of features + label column
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
