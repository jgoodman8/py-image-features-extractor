import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class ClusteringModel(object):
    def __init__(self, min_k: int = 1, max_k: int = 1, step: int = 1, threshold: float = 0.85):
        self.data = None
        self.selected_model = None
        self.models = []
        self.distances = []
        
        self.k = 0
        self.threshold = threshold
        self.k_grid = list(range(min_k, max_k, step))
    
    def add(self, descriptors):
        if self.data is not None:
            self.data = np.concatenate((self.data, descriptors), axis=0)
        else:
            self.data = descriptors
    
    def cluster(self) -> np.ndarray:
        for k in self.k_grid:
            print("Clustering for k: {}".format(k))
            self.cluster_and_evaluate_by_k(k)
        
        self.find_best_model()
        
        return self.selected_model.cluster_centers_
    
    def cluster_and_evaluate_by_k(self, k: int):
        model = KMeans(n_clusters=k).fit(self.data)
        distance = self.evaluate(model)
        
        self.models.append(model)
        self.distances.append(distance)
    
    def evaluate(self, model: KMeans) -> int:
        total_distance = 0
        predicted_clusters = model.fit_predict(self.data)
        
        for sample, cluster_idx in zip(self.data, predicted_clusters):
            centroid = model.cluster_centers_[cluster_idx]
            total_distance += ClusteringModel.get_distance(sample, centroid)
        
        return total_distance
    
    def find_best_model(self):
        
        for idx, dist in enumerate(self.distances[:-1]):
            ratio = dist / self.distances[idx + 1]
            
            if ratio > self.threshold:
                self.k = self.k_grid[idx]
                self.selected_model = self.models[idx]
                break
    
    @staticmethod
    def get_distance(sample: np.ndarray, centroid: np.ndarray) -> int:
        return cdist(np.array([sample]), np.array([centroid]), 'sqeuclidean')[0, 0]
