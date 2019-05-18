import pytest
from click.testing import CliRunner, Result

from image_feature_extractor.cli import main as cli
from image_feature_extractor.tests import utils as test_utils


class TestCli(object):
    
    @pytest.fixture(autouse=True)
    def clear_output_folder(self):
        # No actions before every test
        yield
        test_utils.clean_test_output_csv_route()
    
    @pytest.mark.parametrize('cnn', ['vgg19', 'inception_v3', 'inception_resnet_v2'])
    def test_feature_extraction_with_deep_models(self, cnn: str):
        dst = test_utils.get_test_output_csv_route()
        
        runner = CliRunner()
        result: Result = runner.invoke(cli, ['extract', '--deep', '--cnn', 'vgg19', '--size', 64,
                                             '--src', test_utils.get_test_base_route(), '--dst', dst])
        
        test_utils.assert_validation_test(result)
    
    def test_feature_extraction_with_lbp(self):
        dst = test_utils.get_test_output_csv_route()
        
        runner = CliRunner()
        result: Result = runner.invoke(cli, ['extract', '--lbp', '--size', 64, '--grid', 4, '--points', 8,
                                             '--radius', 1, '--src', test_utils.get_test_base_route(), '--dst', dst])
        
        test_utils.assert_validation_test(result)
    
    @pytest.mark.parametrize('detector', ['kaze', 'sift', 'surf'])
    def test_feature_extraction_with_bow(self, detector: str):
        dst = test_utils.get_test_output_csv_route()
        
        runner = CliRunner()
        result: Result = runner.invoke(cli, ['extract', '--bow', '--size', 64, '--detector', detector,
                                             '--src', test_utils.get_test_base_route(), '--dst', dst, '--k', 2])
        
        test_utils.assert_validation_test(result)
