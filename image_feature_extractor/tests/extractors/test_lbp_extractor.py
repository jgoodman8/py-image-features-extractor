import pytest

from image_feature_extractor.extractors import LBPExtractor
from image_feature_extractor.tests import utils as test_utils


class TestLBPExtractor(object):
    
    @pytest.fixture(autouse=True)
    def clear_output_folder(self):
        # No actions before every test
        yield
        test_utils.clean_test_output_csv_route()
    
    def test_image_is_converted_into_matrix_of_selected_size(self):
        extractor = LBPExtractor(test_utils.get_test_base_route(), size=64, points=3, radius=1, grid_x=2, grid_y=2)
        image = extractor._read_image(test_utils.get_test_image_route())
        
        assert (image.shape == (extractor.width, extractor.height))
    
    def test_local_binary_patterns_are_extracted_from_image(self):
        extractor = LBPExtractor(test_utils.get_test_base_route(), size=64, points=3, radius=1, grid_x=2, grid_y=2)
        image = extractor._read_image(test_utils.get_test_image_route())
        lbp = extractor._extract_lbp(image)
        
        assert (lbp.shape == (extractor.width, extractor.height))
    
    @pytest.mark.parametrize('x', [2, 4])
    @pytest.mark.parametrize('y', [2, 4])
    def test_histogram_is_generated_from_local_binary_patterns(self, x, y):
        extractor = LBPExtractor(test_utils.get_test_base_route(), size=64, points=3, radius=1, grid_x=x, grid_y=y)
        image = extractor._read_image(test_utils.get_test_image_route())
        lbp = extractor._extract_lbp(image)
        histogram = extractor._convert_lbp_to_histogram(lbp)
        
        assert (len(histogram) == x * y)
        
        for histogram_row in histogram:
            assert (histogram_row.shape[0] == len(extractor.bins) - 1)
    
    @pytest.mark.parametrize('x', [2, 4])
    @pytest.mark.parametrize('y', [2, 4])
    def test_single_image_features_extraction(self, x, y):
        extractor = LBPExtractor(test_utils.get_test_base_route(), size=64, points=3, radius=1, grid_x=x, grid_y=y)
        features = extractor.extract(test_utils.get_test_image_route())
        
        expected_size = x * y * (len(extractor.bins) - 1)
        assert (features.shape[0] == expected_size)
    
    @pytest.mark.parametrize('x', [2, 4])
    @pytest.mark.parametrize('y', [2, 4])
    def test_features_are_extracted_from_a_given_image_route(self, x, y):
        output_csv = test_utils.get_test_output_csv_route()
        base_route = test_utils.get_test_base_route()
        
        extractor = LBPExtractor(base_route=base_route, size=64, points=3, radius=1, grid_x=x, grid_y=y)
        extractor.extract_and_save(output_csv)
        
        features = test_utils.load_csv_from_route(output_csv)
        
        expected_width = x * y * (len(extractor.bins) - 1) + 1  # Number of features + label column
        expected_height = test_utils.count_test_images()
        assert (features.shape == (expected_height, expected_width))
