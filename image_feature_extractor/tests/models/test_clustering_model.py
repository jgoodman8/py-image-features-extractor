import numpy as np
import pytest
from sklearn import datasets

from image_feature_extractor.models import ClusteringModel


class TestClusteringModel(object):
    
    def test_data_is_added_to_model(self):
        rows = [np.array([[1, 2, 3], [1, 2, 3]]), np.array([[4, 5, 6], [4, 5, 6]])]
        clustering_model = ClusteringModel()
        
        for row in rows:
            clustering_model.add(row)
        
        assert (np.array_equal(clustering_model.data[0], rows[0][0]))
        assert (np.array_equal(clustering_model.data[1], rows[0][1]))
        assert (np.array_equal(clustering_model.data[2], rows[1][0]))
        assert (np.array_equal(clustering_model.data[3], rows[1][1]))
    
    @pytest.mark.parametrize('k', [2, 3, 4])
    def test_clustering_and_evaluation_for_a_give_k_value(self, k):
        clustering_model = ClusteringModel()
        clustering_model.data = datasets.load_iris().data
        
        clustering_model.cluster_and_evaluate_by_k(k)
        
        assert (len(clustering_model.distances) == 1)
        assert (len(clustering_model.models) == 1)
        assert (clustering_model.models[0].cluster_centers_.shape[0] == k)
    
    def test_model_is_selected_from_a_grid_of_k_values(self):
        clustering_model = ClusteringModel(min_k=2, max_k=10, step=1, threshold=0.9)
        clustering_model.data = datasets.load_iris().data
        
        clustering_model.cluster()
        
        assert (len(clustering_model.distances) == 8)
        assert (len(clustering_model.models) == 8)
        assert (clustering_model.selected_model.cluster_centers_.shape[0] == clustering_model.k)
