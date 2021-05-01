import numpy as np

class KNN():

    def __init__(self, k=3):
        self.k = k
        self.X_train
        self.y_train
        self.labels
    
    # calculate distance between two vectors
    def euclidean_distance(self, v1, v2):
        return np.sqrt(sum((self.v1-self.v2)**2))

    def evaluate(self, y_hat, y_test):
        return sum(y_hat == y_test) / y_hat.shape[0]
    
    
    def fit(self, X_train, y_train):
        # 
        self.X_train = X_train
        self.y_train = y_train
        y_hat = []
        for i, v1 in enumerate(X_test):
            distances = []
            for j, v2 in enumerate(X_train):
                distances.append([j ,euclidean_distance(v1, v2)])
            sorted_dist = sorted(distances, lambda d: d[1])
            neighbors = sorted_dist[:self.k]
            labels = y_train[neighbors[:, 0]]

    # def predict(self, X_test):
  
        
    #     y_pred.append(max(set(labels), key=labels.count))

